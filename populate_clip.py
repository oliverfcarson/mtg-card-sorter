"""
Populate MTG Card Database using CLIP embeddings

CLIP (Contrastive Language-Image Pre-training) produces more semantically
meaningful embeddings than VGG16, which may help distinguish between
similar card printings.
"""

import open_clip
import torch
from PIL import Image, ImageFile
import json
import time
import numpy as np
import os
from multiprocessing import Process, Queue, Value
import chromadb

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_BASE = "./mtg_images"
CHROMA_DB_PATH = "./chroma_db"

# CLIP model to use - ViT-B/32 is a good balance of speed and accuracy
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

num_cores = int(os.cpu_count() * 1)
print("Number of cores:", num_cores)


def worker(in_q, out_q, counter, error_q):
    pid = str(os.getpid())

    print(f"Worker [{pid}] Hello!")

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Worker [{pid}] Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    model = model.to(device)
    model.eval()

    print(f"Worker [{pid}] Loaded CLIP model!")
    worker_c = 0

    print(f"Worker [{pid}] Off we go!")
    while True:
        img_path = in_q.get()
        if img_path is None:
            break

        with counter.get_lock():
            counter.value += 1
        worker_c += 1
        if worker_c % 100 == 0:
            print(f"Worker [{pid}] Processed {worker_c} cards")

        try:
            # Load and preprocess image for CLIP
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            # Get CLIP embedding
            with torch.no_grad():
                embedding = model.encode_image(img_tensor)
                # Normalize the embedding
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                vector = embedding.cpu().numpy().flatten().tolist()

            # Build the Unique ID for the card
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
    print("=" * 50)
    print("MTG Card Database Builder (CLIP)")
    print("=" * 50)
    print(f"\nUsing CLIP model: {CLIP_MODEL} ({CLIP_PRETRAINED})")

    counter = Value('i', 0)
    in_q = Queue()
    out_q = Queue()
    error_q = Queue()
    total = 0
    start_time = time.time()

    worker_count = num_cores * 0
    if worker_count == 0:
        worker_count = 4  # CLIP is heavier, use fewer workers
    print("Worker count:", worker_count)

    # Initialize ChromaDB - use a new collection for CLIP embeddings
    print("\nChecking existing cards in database...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete old CLIP collection if it exists (fresh start)
    try:
        chroma_client.delete_collection(name="mtg_cards_clip")
        print("Deleted existing CLIP collection for fresh start")
    except:
        pass

    collection = chroma_client.create_collection(
        name="mtg_cards_clip",
        metadata={"hnsw:space": "cosine"}
    )
    print("Created new CLIP collection")

    print("\nStarting workers...")
    workers = []
    for i in range(worker_count):
        p = Process(target=worker, args=(in_q, out_q, counter, error_q))
        p.start()
        workers.append(p)

    print("Loading data...")

    all_card_images = []
    for root, dirs, files in os.walk(IMAGE_BASE):
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                all_card_images.append(img_path)

    print(f"Processing {len(all_card_images):,} cards...")

    if len(all_card_images) == 0:
        print("No cards found. Nothing to do.")
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

    print("\nProcessing and storing embeddings...")
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

                if stored_count - last_c >= 500:
                    last_c = stored_count
                    time_remaining = (time.time() - start_time) / stored_count * (total - stored_count)
                    print(f"Stored {stored_count:,} of {total:,} ({stored_count/total * 100:.1f}%) - ETA: {time_remaining / 60:.1f} minutes")

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
        for err in errors[:10]:
            print(f"  - {err['path']}: {err['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print(f"\n{'=' * 50}")
    print(f"All done! Stored {stored_count:,} cards in {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Total cards in CLIP database: {collection.count():,}")
    print(f"{'=' * 50}")
