"""
Build Set Symbol Database for MTG Card Sorter

This script:
1. Fetches all sets from Scryfall API (for metadata)
2. Crops set symbol regions from existing card images
3. Creates VGG16 embeddings for each set symbol
4. Stores them in ChromaDB for fast similarity search
"""

import requests
import os
import json
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
import chromadb
from glob import glob

# ============== CONFIGURATION ==============
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CARD_IMAGES_DIR = os.path.join(SCRIPT_DIR, "mtg_images")
SET_DATA_FILE = os.path.join(SCRIPT_DIR, "data", "mtg-sets.json")
CHROMA_DB_PATH = os.path.join(SCRIPT_DIR, "chroma_db")
SCRYFALL_SETS_URL = "https://api.scryfall.com/sets"

os.makedirs(os.path.join(SCRIPT_DIR, "data"), exist_ok=True)

print("=" * 50)
print("MTG Set Symbol Database Builder")
print("=" * 50)

# ============== STEP 1: FETCH ALL SETS ==============
print("\n[1/4] Fetching sets from Scryfall API...")

response = requests.get(SCRYFALL_SETS_URL)
if response.status_code != 200:
    print(f"Error fetching sets: {response.status_code}")
    exit(1)

sets_data = response.json()
all_sets = sets_data.get('data', [])
print(f"       Found {len(all_sets)} sets!")

# Build set lookup by code
set_lookup = {s['code']: s for s in all_sets}

# Save sets data for reference
with open(SET_DATA_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_sets, f, indent=2)
print(f"       Saved set data to {SET_DATA_FILE}")

# ============== STEP 2: FIND CARD IMAGES BY SET ==============
print("\n[2/4] Finding card images by set...")

# How many sample cards to use per set (more = better matching accuracy)
SAMPLES_PER_SET = 5

# Images are organized in subdirectories by set code: mtg_images/{set_code}/*.jpg
sets_with_images = {}

# Get all set subdirectories
set_dirs = glob(os.path.join(CARD_IMAGES_DIR, "*"))
print(f"       Found {len(set_dirs)} set directories")

for set_dir in set_dirs:
    if os.path.isdir(set_dir):
        set_code = os.path.basename(set_dir)
        # Get up to SAMPLES_PER_SET images from this set
        images_in_set = glob(os.path.join(set_dir, "*.jpg"))
        if images_in_set:
            # Take up to SAMPLES_PER_SET images, spread across the set
            step = max(1, len(images_in_set) // SAMPLES_PER_SET)
            selected_images = images_in_set[::step][:SAMPLES_PER_SET]
            sets_with_images[set_code] = selected_images

total_images = sum(len(imgs) for imgs in sets_with_images.values())
print(f"       Found images for {len(sets_with_images)} unique sets")
print(f"       Using {total_images} total images ({SAMPLES_PER_SET} per set max)")

# ============== STEP 3: LOAD VGG16 MODEL ==============
print("\n[3/4] Loading AI model...")
nn = VGG16(weights='imagenet', include_top=False)
print("       Model loaded!")

# ============== STEP 4: BUILD VECTOR DATABASE ==============
print("\n[4/4] Building set symbol vector database...")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Delete existing collection if it exists
try:
    chroma_client.delete_collection(name="mtg_set_symbols")
    print("       Deleted existing set symbols collection")
except:
    pass

# Create new collection
collection = chroma_client.create_collection(
    name="mtg_set_symbols",
    metadata={"hnsw:space": "cosine"}
)
print("       Created new set symbols collection")


def extract_set_symbol_region(card_image):
    """
    Extract the set symbol region from a card image.
    The set symbol is typically located on the right side, below the card art.
    For modern frames (M15+), it's in the middle-right area.
    """
    height, width = card_image.shape[:2]

    # Set symbol region varies by frame, but generally:
    # - Horizontal: 85-98% from left (right side of card)
    # - Vertical: 55-65% from top (below the art, in the type line area)

    # Extract a region that should contain the set symbol
    x1 = int(width * 0.82)
    x2 = int(width * 0.98)
    y1 = int(height * 0.54)
    y2 = int(height * 0.64)

    symbol_region = card_image[y1:y2, x1:x2]

    return symbol_region


# Process each set (with multiple samples per set)
processed = 0
failed = 0
batch_ids = []
batch_embeddings = []
batch_metadatas = []
BATCH_SIZE = 50

for set_code, img_paths in sets_with_images.items():
    # Process each image for this set
    for img_idx, img_path in enumerate(img_paths):
        try:
            # Load card image
            card_img = cv2.imread(img_path)
            if card_img is None:
                failed += 1
                continue

            # Extract set symbol region
            symbol_region = extract_set_symbol_region(card_img)

            if symbol_region.size == 0:
                failed += 1
                continue

            # Resize to VGG16 input size
            symbol_resized = cv2.resize(symbol_region, (224, 224))
            symbol_rgb = cv2.cvtColor(symbol_resized, cv2.COLOR_BGR2RGB)
            symbol_array = np.expand_dims(symbol_rgb, axis=0).astype('float32')
            x = preprocess_input(symbol_array)

            # Get embedding
            preds = nn.predict(x, verbose=0)
            vector = preds.flatten().tolist()

            # Get set metadata
            set_info = set_lookup.get(set_code, {})

            # Prepare batch data - unique ID for each sample
            sample_id = f"{set_code}_{img_idx}"
            batch_ids.append(sample_id)
            batch_embeddings.append(vector)
            batch_metadatas.append({
                'code': set_code,  # The actual set code for matching
                'name': set_info.get('name', set_code),
                'released_at': set_info.get('released_at', ''),
                'set_type': set_info.get('set_type', ''),
                'card_count': set_info.get('card_count', 0),
                'sample_idx': img_idx
            })

            processed += 1

            # Add batch to database
            if len(batch_ids) >= BATCH_SIZE:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                print(f"       Processed {processed} symbol samples...")
                batch_ids = []
                batch_embeddings = []
                batch_metadatas = []

        except Exception as e:
            failed += 1
            print(f"       Warning: Failed to process {set_code} sample {img_idx}: {e}")

# Add remaining items
if batch_ids:
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )

print(f"\n{'=' * 50}")
print(f"Set symbol database built successfully!")
print(f"Total sets processed: {processed}")
print(f"Failed: {failed}")
print(f"Database location: {CHROMA_DB_PATH}")
print(f"{'=' * 50}")
