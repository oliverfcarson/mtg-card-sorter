#from pymilvus import MilvusClient
#import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import json
import time
import requests
import numpy as np
from keras.preprocessing import image
import os
from multiprocessing import Process, Queue, Value
import chromadb
from PIL import ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_BASE = "./mtg_images"
CHROMA_DB_PATH = "./chroma_db"

#print(tf.config.list_physical_devices())
num_cores = int(os.cpu_count() * 1) #1 = 52, 0.5 = 57, 1.5 = 60
print("Number of cores:", num_cores)


def worker(in_q, out_q, counter, error_q):
  pid = str(os.getpid())

  print(f"Worker [{pid}] Hello!")
  nn = VGG16(weights='imagenet',  include_top=False)
  print(f"Worker [{pid}] Loaded model!")
  worker_c = 0

  print(f"Worker [{pid}] Off we go!")
  while True:
    img_path = in_q.get()
    if img_path == None:
      break

    with counter.get_lock():
      counter.value += 1
    worker_c += 1
    if worker_c % 100 == 0:
      print(f"Worker [{pid}] Processed {worker_c} cards")

    try:
      img = image.load_img(img_path, target_size=(224, 224))
      img = image.img_to_array(img)
      #Keras, the toolkit we're using, needs to preprocess the image.
      x = preprocess_input(np.expand_dims(img.copy(), axis=0))
      #Finally run it through VGG16 to get a vector
      preds = nn.predict(x, verbose=None)
      #Reformat the vector so it's in a good format
      vector = preds.flatten().tolist()
      #Build the Unique ID for the card
      card_set = img_path.split(os.path.sep)[-2]
      card_num = img_path.split(os.path.sep)[-1].split(".")[0]
      card_id = f"{card_set}:{card_num}"

      # Send result to main process for DB storage
      out_q.put({
          "id": card_id,
          "embedding": vector,
          "metadata": {"set": card_set, "num": card_num}
      })
    except Exception as e:
      error_q.put({"path": img_path, "error": str(e)})

if __name__ == '__main__':
  counter = Value('i', 0)
  in_q = Queue()
  out_q = Queue()
  error_q = Queue()
  total = 0
  start_time = time.time()

  worker_count = num_cores * 0
  if worker_count == 0:
    worker_count = 8
  print("Worker count:", worker_count)

  # Initialize ChromaDB early to check existing cards
  print("Checking existing cards in database...")
  chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
  collection = chroma_client.get_or_create_collection(
      name="mtg_cards",
      metadata={"hnsw:space": "cosine"}
  )

  # Get existing IDs to skip
  existing_count = collection.count()
  print(f"Found {existing_count:,} cards already in database")

  existing_ids = set()
  if existing_count > 0:
    # Fetch existing IDs in batches
    print("Loading existing IDs for resume...")
    batch_size = 10000
    for offset in range(0, existing_count, batch_size):
      result = collection.get(limit=batch_size, offset=offset, include=[])
      existing_ids.update(result['ids'])
    print(f"Loaded {len(existing_ids):,} existing IDs")

  print("Starting workers...")
  workers = []
  for i in range(worker_count):
    p = Process(target=worker, args=(in_q, out_q, counter, error_q))
    p.start()
    workers.append(p)

  print("Loading data...")

  all_card_images = []
  skipped = 0
  for root, dirs, files in os.walk(IMAGE_BASE):
    for file in files:
      if file.endswith(".jpg"):
        img_path = os.path.join(root, file)
        # Build ID to check if already exists
        card_set = img_path.split(os.path.sep)[-2]
        card_num = file.split(".")[0]
        card_id = f"{card_set}:{card_num}"

        if card_id in existing_ids:
          skipped += 1
        else:
          all_card_images.append(img_path)

  print(f"Skipped {skipped:,} cards already in database")
  print(f"Processing {len(all_card_images):,} new cards...")

  if len(all_card_images) == 0:
    print("All cards already processed. Nothing to do.")
    for _ in range(worker_count):
      in_q.put(None)
    exit()

  for card_image in all_card_images:
    in_q.put(card_image)

  # Send stop signals to workers
  for _ in range(worker_count):
    in_q.put(None)

  total = len(all_card_images)

  # Collect results and batch insert to ChromaDB
  batch_size = 100
  batch_ids = []
  batch_embeddings = []
  batch_metadatas = []
  stored_count = 0
  last_c = 0

  print("Processing and storing embeddings...")
  while stored_count < total:
    try:
      result = out_q.get(timeout=1)
      batch_ids.append(result["id"])
      batch_embeddings.append(result["embedding"])
      batch_metadatas.append(result["metadata"])

      if len(batch_ids) >= batch_size:
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
        stored_count += len(batch_ids)
        batch_ids = []
        batch_embeddings = []
        batch_metadatas = []

        if stored_count - last_c >= 100:
          last_c = stored_count
          time_remaining = (time.time() - start_time) / stored_count * (total - stored_count)
          print(f"Stored {stored_count} of {total} ({stored_count/total * 100:.1f}%) - ETA: {time_remaining / 60:.1f} minutes")

    except:
      pass

  # Store remaining batch
  if batch_ids:
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )
    stored_count += len(batch_ids)

  # Wait for workers to finish
  for p in workers:
    p.join()

  # Report any errors
  errors = []
  while not error_q.empty():
    errors.append(error_q.get())

  if errors:
    print(f"\nWarning: {len(errors)} images failed to process:")
    for err in errors[:10]:  # Show first 10
      print(f"  - {err['path']}: {err['error']}")
    if len(errors) > 10:
      print(f"  ... and {len(errors) - 10} more")

  print(f"\nAll done! Stored {stored_count} new cards in {(time.time() - start_time) / 60:.1f} minutes")
  print(f"Total cards in database: {collection.count():,}")