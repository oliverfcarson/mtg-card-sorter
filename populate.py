from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image
import os
import chromadb

#Vector dimension for VGG16, it's how many values are in our output vector
# How did I know this? I just have VGG16 spit out a vector and check the len() of it
VECTOR_DIM = 25088
IMAGE_BASE = "./mtg_images"
CHROMA_DB_PATH = "./chroma_db"

#Load the VGG16 model
nn = VGG16(weights='imagenet',  include_top=False)

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="mtg_cards",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

# Let's find all the images we previously downloaded
all_card_images = []
for root, dirs, files in os.walk(IMAGE_BASE):
    for file in files:
        if file.endswith(".jpg"):
            all_card_images.append(os.path.join(root, file))


current = 0
total = len(all_card_images)
batch_size = 100
batch_ids = []
batch_embeddings = []
batch_metadatas = []

#Loop through them to put them in the DB
for card_image in all_card_images:
    current += 1
    if current % 100 == 0:
        print(f"Processing image {current} of {total}   {current/total * 100:.1f}% ...")

    #Load the image and resize it to 224x224, this is the only size VGG16 can take
    img = image.load_img(card_image, target_size=(224, 224))
    img = image.img_to_array(img)
    #Keras, the toolkit we're using, needs to preprocess the image.
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    #Finally run it through VGG16 to get a vector
    preds = nn.predict(x, verbose=None)
    #Reformat the vector so it's in a good format
    vector = preds.flatten().tolist()

    #Build the Unique ID for the card
    card_set = card_image.split(os.path.sep)[-2]
    card_num = card_image.split(os.path.sep)[-1].split(".")[0]
    card_id = f"{card_set}:{card_num}"

    # Add to batch
    batch_ids.append(card_id)
    batch_embeddings.append(vector)
    batch_metadatas.append({"set": card_set, "num": card_num})

    # Store in batches for efficiency
    if len(batch_ids) >= batch_size:
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
        batch_ids = []
        batch_embeddings = []
        batch_metadatas = []

# Store any remaining cards
if batch_ids:
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )

print(f"Done! Added {total} cards to ChromaDB.")