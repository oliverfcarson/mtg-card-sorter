"""
Crop Set Symbols from MTG Card Images

This script extracts the set symbol region from all card images
and organizes them into folders by set code for CNN training.
"""

import os
import cv2
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============== CONFIGURATION ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARD_IMAGES_DIR = os.path.join(SCRIPT_DIR, "mtg_images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "set_symbols_cropped")

# Minimum images per set to include in training
MIN_IMAGES_PER_SET = 10

print("=" * 50)
print("MTG Set Symbol Cropper")
print("=" * 50)


def extract_set_symbol_region(card_image):
    """
    Extract the set symbol region from a card image.
    The set symbol is located on the right side, in the type line area.
    """
    height, width = card_image.shape[:2]

    # Set symbol region:
    # - Horizontal: 82-98% from left (right side of card)
    # - Vertical: 54-64% from top (in the type line area)
    x1 = int(width * 0.82)
    x2 = int(width * 0.98)
    y1 = int(height * 0.54)
    y2 = int(height * 0.64)

    symbol_region = card_image[y1:y2, x1:x2]
    return symbol_region


def process_card(img_path, output_dir):
    """Process a single card image and save the cropped set symbol."""
    try:
        # Load card image
        card_img = cv2.imread(img_path)
        if card_img is None:
            return None

        # Extract set symbol region
        symbol_region = extract_set_symbol_region(card_img)

        if symbol_region.size == 0 or symbol_region.shape[0] < 10 or symbol_region.shape[1] < 10:
            return None

        # Get filename from path
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)

        # Save the cropped symbol
        cv2.imwrite(output_path, symbol_region)
        return output_path

    except Exception as e:
        return None


# ============== STEP 1: FIND ALL CARD IMAGES ==============
print("\n[1/3] Finding card images...")

set_dirs = [d for d in glob(os.path.join(CARD_IMAGES_DIR, "*")) if os.path.isdir(d)]
print(f"       Found {len(set_dirs)} set directories")

# Build list of all images organized by set
sets_with_images = {}
for set_dir in set_dirs:
    set_code = os.path.basename(set_dir)
    images = glob(os.path.join(set_dir, "*.jpg"))
    if len(images) >= MIN_IMAGES_PER_SET:
        sets_with_images[set_code] = images

total_images = sum(len(imgs) for imgs in sets_with_images.values())
print(f"       Found {len(sets_with_images)} sets with >= {MIN_IMAGES_PER_SET} images")
print(f"       Total images to process: {total_images:,}")

# ============== STEP 2: CREATE OUTPUT DIRECTORIES ==============
print("\n[2/3] Creating output directories...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for set_code in sets_with_images.keys():
    set_output_dir = os.path.join(OUTPUT_DIR, set_code)
    os.makedirs(set_output_dir, exist_ok=True)

print(f"       Created {len(sets_with_images)} set directories in {OUTPUT_DIR}")

# ============== STEP 3: CROP SET SYMBOLS ==============
print("\n[3/3] Cropping set symbols...")

processed = 0
failed = 0
NUM_WORKERS = 8

for set_code, img_paths in sets_with_images.items():
    set_output_dir = os.path.join(OUTPUT_DIR, set_code)
    set_processed = 0
    set_failed = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_card, img_path, set_output_dir): img_path
            for img_path in img_paths
        }

        for future in as_completed(futures):
            result = future.result()
            if result:
                set_processed += 1
            else:
                set_failed += 1

    processed += set_processed
    failed += set_failed

    if (processed + failed) % 1000 == 0:
        print(f"       Progress: {processed + failed:,} / {total_images:,}")

# ============== SUMMARY ==============
print(f"\n{'=' * 50}")
print(f"Set symbol cropping complete!")
print(f"Successfully processed: {processed:,}")
print(f"Failed: {failed:,}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'=' * 50}")

# Show stats per set
print("\nSets with most images:")
set_counts = []
for set_code in sets_with_images.keys():
    set_dir = os.path.join(OUTPUT_DIR, set_code)
    count = len(glob(os.path.join(set_dir, "*.jpg")))
    if count > 0:
        set_counts.append((set_code, count))

set_counts.sort(key=lambda x: x[1], reverse=True)
for set_code, count in set_counts[:10]:
    print(f"  {set_code}: {count} images")

print(f"\nTotal sets ready for training: {len([c for c in set_counts if c[1] >= MIN_IMAGES_PER_SET])}")
