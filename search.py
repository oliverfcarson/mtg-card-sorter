from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image
import chromadb

CHROMA_DB_PATH = "./chroma_db"

#Load the VGG16 model
nn = VGG16(weights='imagenet',  include_top=False)

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name="mtg_cards")

img = image.load_img("card_image.jpg", target_size=(224, 224))

img = image.img_to_array(img)
x = preprocess_input(np.expand_dims(img.copy(), axis=0))
print("Sending image to model")
preds = nn.predict(x, verbose=None)
vector = preds.flatten().tolist()

# Query ChromaDB for similar cards
results = collection.query(
    query_embeddings=[vector],
    n_results=5
)

# Display results
for i in range(len(results['ids'][0])):
    card_id = results['ids'][0][i]
    metadata = results['metadatas'][0][i]
    distance = results['distances'][0][i]
    # ChromaDB returns cosine distance, convert to similarity score
    score = 1 - distance
    print(f"{i}. {metadata['set']} {metadata['num']} (Score: {round(score, 3)})")