import requests
import json
import os

source_json = "./data/mtg-default-cards.json"
IMAGE_BASE = "./mtg_images"

print(f"Loading JSON from {source_json}...")
print("(This may take a moment - the file is large)")

with open(source_json, encoding='utf-8') as f:
    json_data = json.load(f)

print(f"Loaded {len(json_data):,} total cards from JSON")
print("Filtering out cards with missing images...")

# First pass: count matching cards and skip ones we already have
cards_to_download = []
skipped_existing = 0
skipped_missing = 0

for card in json_data:
    # Skip cards with no image
    if card.get("image_status") == "missing":
        skipped_missing += 1
        continue

    # Check if we already downloaded this card
    card_set = card.get("set", "unknown")
    card_collectors_number = card.get("collector_number", "0")
    card_path = os.path.join(IMAGE_BASE, card_set, f"{card_collectors_number}.jpg")

    if os.path.exists(card_path):
        skipped_existing += 1
        continue

    cards_to_download.append(card)

total_to_download = len(cards_to_download)
print(f"Skipped {skipped_missing:,} cards with missing images")
print(f"Skipped {skipped_existing:,} cards already downloaded")
print(f"Found {total_to_download:,} cards to download")

if total_to_download == 0:
    print("All cards already downloaded. Nothing to do.")
    exit()

print("---------------------------------------------------")
print("Starting downloads... (Press Ctrl+C to pause - you can resume later)")
print("---------------------------------------------------")

downloaded = 0
failed = 0

try:
    for card in cards_to_download:
        downloaded += 1
        card_name = card.get("name", "Unknown")
        card_set = card.get("set", "unknown")
        card_collectors_number = card.get("collector_number", "0")

        if downloaded % 100 == 0 or downloaded <= 5:
            print(f"[{downloaded:,}/{total_to_download:,}] Downloading {card_name} ({card_set})...")

        try:
            try:
                card_image_url = card["image_uris"]["normal"]
            except:
                card_image_url = card["card_faces"][0]["image_uris"]["normal"]  # Double-faced cards

            card_image_path = os.path.join(IMAGE_BASE, card_set)
            os.makedirs(card_image_path, exist_ok=True)

            card_file_name = f"{card_collectors_number}.jpg"
            full_path = os.path.join(card_image_path, card_file_name)

            response = requests.get(card_image_url, timeout=30)
            response.raise_for_status()

            with open(full_path, "wb") as file:
                file.write(response.content)

        except Exception as e:
            failed += 1
            print(f"    Failed to download {card_name}: {e}")

except KeyboardInterrupt:
    print("\n---------------------------------------------------")
    print(f"Paused! Downloaded {downloaded:,} cards so far.")
    print("Run the script again to resume from where you left off.")
    exit()

print("---------------------------------------------------")
print(f"Done! Downloaded {downloaded:,} cards ({failed} failed)")
print(f"Images saved to {IMAGE_BASE}/")