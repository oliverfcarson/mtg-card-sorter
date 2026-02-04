import cv2
import numpy as np
from itertools import combinations
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import chromadb
from threading import Thread
from queue import Queue
import time
import json
import csv
import os
from datetime import datetime

print("=" * 50)
print("MTG Rapid Card Scanner")
print("=" * 50)

# ============== LOAD CARD DATA ==============
CARD_DATA_PATH = "./data/mtg-default-cards.json"
card_lookup = {}

print("\n[1/4] Loading card database...")
if os.path.exists(CARD_DATA_PATH):
    print("       Loading card data JSON (this may take a moment)...")
    with open(CARD_DATA_PATH, encoding='utf-8') as f:
        card_data = json.load(f)

    # Build lookup by set:collector_number
    for card in card_data:
        set_code = card.get("set", "")
        collector_num = card.get("collector_number", "")
        key = f"{set_code}:{collector_num}"
        card_lookup[key] = {
            "name": card.get("name", "Unknown"),
            "set_code": set_code,
            "set_name": card.get("set_name", "Unknown Set"),
            "collector_number": collector_num,
            "rarity": card.get("rarity", "unknown"),
            "type_line": card.get("type_line", ""),
            "mana_cost": card.get("mana_cost", ""),
            "oracle_text": card.get("oracle_text", ""),
            "prices_usd": card.get("prices", {}).get("usd", ""),
            "prices_usd_foil": card.get("prices", {}).get("usd_foil", ""),
        }
    print(f"       Loaded {len(card_lookup):,} cards into lookup table!")
    del card_data  # Free memory
else:
    print(f"       Warning: Card data not found at {CARD_DATA_PATH}")
    print("       CSV export will have limited information")

# ============== LOAD MODEL & DATABASE ==============
print("\n[2/4] Loading VGG16 model...")

# Load VGG16 for feature extraction (include_top=False gives 25088-dim vectors to match database)
vgg_model = VGG16(weights='imagenet', include_top=False)
print("       VGG16 model loaded!")

CHROMA_DB_PATH = "./chroma_db"
print("       Connecting to card database...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = chroma_client.get_collection(name="mtg_cards")
print(f"       Card database connected! ({collection.count():,} cards)")

# Load CNN set classifier
print("       Loading set symbol CNN classifier...")
SET_CLASSIFIER_PATH = "./models/set_classifier_latest.pth"
set_classifier = None
set_idx_to_class = None
SET_CNN_ENABLED = False

# Define device and transform at module level
cnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if os.path.exists(SET_CLASSIFIER_PATH):
    try:
        # Load model checkpoint
        checkpoint = torch.load(SET_CLASSIFIER_PATH, map_location=cnn_device, weights_only=False)

        num_classes = checkpoint['num_classes']
        set_idx_to_class = checkpoint['idx_to_class']

        # Create model architecture
        set_classifier = models.resnet18(weights=None)
        set_classifier.fc = nn.Linear(set_classifier.fc.in_features, num_classes)
        set_classifier.load_state_dict(checkpoint['model_state_dict'])
        set_classifier = set_classifier.to(cnn_device)
        set_classifier.eval()

        print(f"       Set CNN loaded! ({num_classes} classes, using {cnn_device})")
        if checkpoint.get('best_val_acc'):
            print(f"       Model validation accuracy: {checkpoint['best_val_acc']:.1f}%")
        SET_CNN_ENABLED = True
    except Exception as e:
        print(f"       Warning: Failed to load set CNN: {e}")
        print("       Continuing without set symbol classification...")
else:
    print("       Warning: Set classifier not found. Run train_set_classifier.py first.")
    print("       Continuing without set symbol classification...")

def get_card_info(set_code, collector_num):
    """Look up full card info from the JSON data"""
    key = f"{set_code}:{collector_num}"
    if key in card_lookup:
        return card_lookup[key]
    return {
        "name": "Unknown",
        "set_code": set_code,
        "set_name": "Unknown Set",
        "collector_number": collector_num,
        "rarity": "unknown",
        "type_line": "",
        "mana_cost": "",
        "oracle_text": "",
        "prices_usd": "",
        "prices_usd_foil": "",
    }

# ============== IMAGE PROCESSING FUNCTIONS ==============
def enhance_card_image(image):
    """Enhance card image for better matching with aggressive sharpening"""
    # Step 1: CLAHE for contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Step 2: Unsharp mask for aggressive sharpening (brings out text/details)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)

    # Step 3: Additional high-pass sharpening for fine details (collector numbers, set symbols)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(sharpened, -1, kernel)

    # Blend: 70% sharpened, 30% enhanced (mostly sharp, some smoothness to reduce noise)
    result = cv2.addWeighted(enhanced, 0.3, sharpened, 0.7, 0)
    return result

def calculate_sharpness(image):
    """Calculate image sharpness using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def extract_set_symbol_region(card_image):
    """
    Extract the set symbol region from a card image.
    The set symbol is located on the right side, in the type line area.
    """
    height, width = card_image.shape[:2]

    # Set symbol region (same as in build_set_database.py):
    # - Horizontal: 82-98% from left (right side of card)
    # - Vertical: 54-64% from top (in the type line area)
    x1 = int(width * 0.82)
    x2 = int(width * 0.98)
    y1 = int(height * 0.54)
    y2 = int(height * 0.64)

    symbol_region = card_image[y1:y2, x1:x2]
    return symbol_region


def classify_set_symbol(card_image, top_n=10):
    """
    Classify the set symbol using the trained CNN.
    Returns list of (set_code, confidence) tuples sorted by confidence descending.
    """
    if not SET_CNN_ENABLED or set_classifier is None:
        return []

    try:
        # Extract set symbol region
        symbol_region = extract_set_symbol_region(card_image)

        if symbol_region.size == 0:
            return []

        # Convert to PIL Image for PyTorch
        symbol_rgb = cv2.cvtColor(symbol_region, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(symbol_rgb)

        # Preprocess for CNN
        img_tensor = set_cnn_transform(pil_image).unsqueeze(0).to(cnn_device)

        # Get CNN predictions
        with torch.no_grad():
            outputs = set_classifier(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get top N predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_n, probabilities.size(1)))

            matches = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                set_code = set_idx_to_class[idx.item()]
                confidence = prob.item()
                matches.append((set_code, confidence))

            return matches
    except Exception as e:
        print(f"       Warning: CNN classification error: {e}")
        return []


# Keep old function name for compatibility
def match_set_symbol(card_image, top_n=10):
    """Wrapper for backward compatibility - now uses CNN classification."""
    return classify_set_symbol(card_image, top_n)


def refine_with_set_symbol(card_image, vgg_results, card_lookup):
    """
    Refine VGG16 card results using CNN set symbol classification.
    Strategy:
    1. Get CNN predictions for the set symbol
    2. Look for VGG16 candidates that match a predicted set
    3. Boost candidates that match both card appearance and set prediction
    """
    # Get set symbol matches
    set_matches = match_set_symbol(card_image, top_n=10)

    if not set_matches:
        # No set symbol matching available, return original top result
        top_match = vgg_results['metadatas'][0][0]
        top_score = 1 - vgg_results['distances'][0][0]
        return top_match, top_score, None

    # Create a set of likely set codes from set symbol matching
    likely_sets = {code: score for code, score in set_matches}

    # Score each VGG16 candidate based on combined card + set symbol match
    best_match = None
    best_combined_score = -1
    set_match_info = None

    for i in range(len(vgg_results['ids'][0])):
        meta = vgg_results['metadatas'][0][i]
        distance = vgg_results['distances'][0][i]
        card_score = 1 - distance

        candidate_set = meta['set']

        # Check if this candidate's set matches a top set symbol
        if candidate_set in likely_sets:
            set_score = likely_sets[candidate_set]
            # Combined score: weight card match higher (70%) than set symbol (30%)
            combined_score = (card_score * 0.7) + (set_score * 0.3)

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_match = meta
                set_match_info = {
                    'set_code': candidate_set,
                    'set_score': set_score,
                    'card_score': card_score,
                    'combined_score': combined_score
                }

    # If we found a match with set symbol confirmation, return it
    if best_match is not None:
        return best_match, best_combined_score, set_match_info

    # No set symbol match in VGG16 candidates, return original top result
    top_match = vgg_results['metadatas'][0][0]
    top_score = 1 - vgg_results['distances'][0][0]
    return top_match, top_score, None


def extract_card_from_frame(frame, marker_bounding_boxes):
    """Extract and process card image from frame with ArUco markers"""
    # Process ArUco markers
    centers = []
    max_width = 0
    max_height = 0

    for box in marker_bounding_boxes:
        box_corners = box[0]
        center_x = int(box_corners[:, 0].mean())
        center_y = int(box_corners[:, 1].mean())
        width = int(box_corners[:, 0].max() - box_corners[:, 0].min())
        height = int(box_corners[:, 1].max() - box_corners[:, 1].min())
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height
        centers.append((center_x, center_y))

    min_x = min(centers, key=lambda x: x[0])[0] + max_width / 2
    max_x = max(centers, key=lambda x: x[0])[0] - max_width / 2
    min_y = min(centers, key=lambda x: x[1])[1] + max_height / 2
    max_y = max(centers, key=lambda x: x[1])[1] - max_height / 2

    centers_np = np.array([
        [min_x, min_y], [max_x, min_y],
        [max_x, max_y], [min_x, max_y]
    ], dtype="float32")

    rect = cv2.convexHull(centers_np)
    rect = np.array(rect, dtype="float32")

    width = int(max_x - min_x)
    height = int(max_y - min_y)
    dst = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    aruco_cropped = cv2.warpPerspective(frame, M, (width, height))

    # Detect card edges
    gray = cv2.cvtColor(aruco_cropped, cv2.COLOR_BGR2GRAY)

    best_contours = []
    for method_func in [
        lambda g: cv2.Canny(cv2.GaussianBlur(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2), (5,5), 0), 30, 100),
        lambda g: cv2.Canny(cv2.GaussianBlur(g, (5,5), 0), 50, 150),
        lambda g: cv2.Canny(cv2.threshold(cv2.GaussianBlur(g, (5,5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 50, 150),
    ]:
        e = method_func(gray)
        c, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(c) > len(best_contours):
            best_contours = c

    if len(best_contours) == 0:
        return None

    # Find card contour
    image_area = aruco_cropped.shape[0] * aruco_cropped.shape[1]
    min_card_area = image_area * 0.05
    sorted_contours = sorted(best_contours, key=cv2.contourArea, reverse=True)

    best_contour = None
    for contour in sorted_contours[:10]:
        area = cv2.contourArea(contour)
        if area < min_card_area:
            continue
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx_poly = cv2.approxPolyDP(contour, epsilon, True)
        if 4 <= len(approx_poly) <= 6:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = min(w, h) / max(w, h)
            if 0.5 <= aspect_ratio <= 0.9:
                best_contour = contour
                break

    if best_contour is None:
        best_contour = sorted_contours[0]

    largest_contour = best_contour
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Find 4 corners
    max_area = 0
    best_quad = None

    if len(approx) < 4:
        rect_fit = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect_fit)
        best_quad = np.array(box, dtype="float32").reshape(4, 1, 2)
    else:
        for quad in combinations(approx, 4):
            quad = np.array(quad, dtype="float32")
            area = cv2.contourArea(quad)
            if area > max_area:
                max_area = area
                best_quad = quad

    if best_quad is None:
        return None

    points = np.squeeze(best_quad)
    if points.ndim == 1:
        points = points.reshape(4, 2)

    # Sort points clockwise
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    top_width = rect[1][0] - rect[0][0]
    bottom_width = rect[3][0] - rect[2][0]
    left_height = rect[0][1] - rect[3][1]
    right_height = rect[2][1] - rect[1][1]

    width = int(abs(max(top_width, bottom_width)))
    height = int(abs(max(left_height, right_height)))

    if width < 50 or height < 50:
        return None

    dst = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    card_image = cv2.warpPerspective(aruco_cropped, matrix, (width, height))

    # Enhance the card
    card_image = enhance_card_image(card_image)

    return card_image

# ============== ASYNC SEARCH WORKER ==============
search_queue = Queue()
results_list = []
worker_running = True

# Confidence threshold for auto-retry
CONFIDENCE_THRESHOLD = 0.70  # Below this, we use multi-capture consensus
MULTI_CAPTURE_COUNT = 5      # Number of frames to capture for consensus


def search_single_image(card_image):
    """Search a single card image and return (match, score, set_info, all_results)"""
    # Resize and preprocess for VGG16
    img_resized = cv2.resize(card_image, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    img_preprocessed = preprocess_input(img_array.astype('float32'))

    # Get VGG16 embedding (25088 dimensions)
    vector = vgg_model.predict(img_preprocessed, verbose=0).flatten().tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[vector],
        n_results=15
    )

    # Refine with CNN set classification
    if SET_CNN_ENABLED:
        top_match, top_score, set_info = refine_with_set_symbol(card_image, results, card_lookup)
    else:
        top_match = results['metadatas'][0][0] if results['metadatas'][0] else None
        top_score = 1 - results['distances'][0][0] if results['distances'][0] else 0
        set_info = None

    return top_match, top_score, set_info, results


def consensus_vote(search_results):
    """
    Use consensus voting from multiple search results.
    Returns the best match based on voting + score weighting.

    search_results: list of (match, score, set_info, all_results) tuples
    """
    if not search_results:
        return None, 0, None, None

    if len(search_results) == 1:
        match, score, set_info, all_results = search_results[0]
        return match, score, set_info, all_results

    # Count votes for each unique card (set:num combination)
    votes = {}  # key: "set:num", value: {'count': int, 'scores': [], 'matches': [], 'set_infos': [], 'all_results': []}

    for match, score, set_info, all_results in search_results:
        if match is None:
            continue
        key = f"{match['set']}:{match['num']}"
        if key not in votes:
            votes[key] = {'count': 0, 'scores': [], 'matches': [], 'set_infos': [], 'all_results': []}
        votes[key]['count'] += 1
        votes[key]['scores'].append(score)
        votes[key]['matches'].append(match)
        votes[key]['set_infos'].append(set_info)
        votes[key]['all_results'].append(all_results)

    if not votes:
        # All searches failed
        return None, 0, None, None

    # Find the best candidate:
    # 1. Prefer candidates with more votes
    # 2. Among ties, prefer higher average score
    best_key = None
    best_vote_count = 0
    best_avg_score = 0

    for key, data in votes.items():
        avg_score = sum(data['scores']) / len(data['scores'])
        if data['count'] > best_vote_count or (data['count'] == best_vote_count and avg_score > best_avg_score):
            best_key = key
            best_vote_count = data['count']
            best_avg_score = avg_score

    # Return the best result
    best_data = votes[best_key]
    # Use the highest-scoring instance of this card
    best_idx = best_data['scores'].index(max(best_data['scores']))

    return (
        best_data['matches'][best_idx],
        max(best_data['scores']),  # Use best score
        best_data['set_infos'][best_idx],
        best_data['all_results'][best_idx]
    )


def search_worker():
    """Background worker that processes cards with multi-capture consensus"""
    global results_list, worker_running

    while worker_running:
        try:
            item = search_queue.get(timeout=0.5)
            if item is None:
                break

            scan_num, card_images, thumbnail = item

            # card_images can be a single image or a list of images
            if not isinstance(card_images, list):
                card_images = [card_images]

            # Always use consensus when multiple images available for best accuracy
            if len(card_images) == 1:
                # Only one image, search it directly
                top_match, top_score, set_info, all_results = search_single_image(card_images[0])
                consensus_used = False
            else:
                # Multiple images - always use consensus voting
                print(f"       [Scan #{scan_num}] Searching {len(card_images)} frames with consensus...")
                search_results = []

                for img in card_images:
                    match, score, s_info, results = search_single_image(img)
                    search_results.append((match, score, s_info, results))

                # Use consensus voting
                top_match, top_score, set_info, all_results = consensus_vote(search_results)
                consensus_used = True

            if top_match is None:
                print(f"       [Scan #{scan_num}] Search failed")
                continue

            results_list.append({
                'scan_num': scan_num,
                'thumbnail': thumbnail,
                'top_match': top_match,
                'top_score': top_score,
                'all_results': all_results,
                'set_info': set_info,
                'consensus_used': consensus_used if 'consensus_used' in dir() else False
            })

            # Look up card name for display
            card_info = get_card_info(top_match['set'], top_match['num'])
            set_str = ""
            if set_info:
                set_str = f" [Set: {set_info['set_code']} ({set_info['set_score']:.2f})]"
            consensus_str = " [CONSENSUS]" if consensus_used else ""
            print(f"       [Scan #{scan_num}] Found: {card_info['name']} ({top_match['set']} #{top_match['num']}) Score: {top_score:.3f}{set_str}{consensus_str}")

        except Exception as e:
            if str(e):  # Only print if there's an actual error message
                print(f"       [Worker] Error: {e}")

# Start the search worker thread
search_thread = Thread(target=search_worker, daemon=True)
search_thread.start()

# ============== CAMERA SETUP ==============
print("\n[3/4] Opening webcam...")

CAMERA = None
camera_index = None

for idx in range(5):
    print(f"       Trying camera index {idx}...")
    cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cam.isOpened():
        ret, test_frame = cam.read()
        if ret and test_frame is not None:
            print(f"       Found working camera at index {idx}!")
            CAMERA = cam
            camera_index = idx
            break
        else:
            cam.release()
    else:
        cam.release()

if CAMERA is None:
    print("\nERROR: Could not find any working camera!")
    exit(1)

CAMERA.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
CAMERA.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
CAMERA.set(cv2.CAP_PROP_AUTOFOCUS, 1)

current_focus = 0
manual_focus_mode = False

print(f"       Webcam opened successfully!")

# ============== ARUCO SETUP ==============
print("\n[4/4] Ready for rapid scanning!")
print("=" * 50)
print("Controls:")
print("  SPACE  = Capture card manually")
print("  A      = Toggle auto-capture")
print("  R      = Show results and exit")
print("  Q      = Quit without results")
print("  F      = Toggle focus mode")
print("  +/-    = Adjust manual focus")
print("=" * 50)

ARUCO_DICTS = {
    1: ("4X4_50", cv2.aruco.DICT_4X4_50),
    2: ("4X4_100", cv2.aruco.DICT_4X4_100),
    3: ("5X5_50", cv2.aruco.DICT_5X5_50),
    4: ("6X6_50", cv2.aruco.DICT_6X6_50),
    5: ("ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
}

current_dict_id = 4
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[current_dict_id][1])
parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshConstant = 7
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.minMarkerPerimeterRate = 0.03
parameters.maxMarkerPerimeterRate = 4.0
parameters.polygonalApproxAccuracyRate = 0.03
parameters.minCornerDistanceRate = 0.05
parameters.minDistanceToBorder = 3

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ============== RAPID SCAN LOOP ==============
scan_count = 0
auto_capture_enabled = True
AUTO_CAPTURE_THRESHOLD = 50  # Auto-capture when sharpness reaches this level
STABLE_FRAMES_REQUIRED = 10
COOLDOWN_FRAMES = 30  # Frames to wait after capture before allowing another

sharpness_history = []
MAX_HISTORY = 60
peak_detected = False
frames_since_peak = 0
best_frame = None
best_sharpness = 0
best_markers = None
cooldown_counter = 0

show_results = False

while True:
    returncode, frame = CAMERA.read()
    if not returncode:
        continue

    # Cooldown after capture
    if cooldown_counter > 0:
        cooldown_counter -= 1

    # Calculate sharpness
    sharpness = calculate_sharpness(frame)
    sharpness_history.append(sharpness)
    if len(sharpness_history) > MAX_HISTORY:
        sharpness_history.pop(0)

    # Detect markers
    marker_bounding_boxes, ids, rejected = detector.detectMarkers(frame)

    # Draw preview
    preview = frame.copy()

    # Marker status
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(preview, marker_bounding_boxes, ids)
        marker_status = f"Markers: {len(ids)}/4"
        marker_color = (0, 255, 0) if len(ids) == 4 else (0, 165, 255)
    else:
        marker_status = "Markers: 0/4"
        marker_color = (0, 0, 255)

    # Sharpness bar
    sharpness_normalized = min(sharpness / 200, 1.0)
    bar_width = int(300 * sharpness_normalized)
    bar_color = (0, 255, 0) if sharpness > AUTO_CAPTURE_THRESHOLD else (0, 165, 255)
    cv2.rectangle(preview, (10, 180), (310, 210), (50, 50, 50), -1)
    cv2.rectangle(preview, (10, 180), (10 + bar_width, 210), bar_color, -1)
    cv2.putText(preview, f"Sharpness: {int(sharpness)}", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Status
    cv2.putText(preview, marker_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, marker_color, 2)
    cv2.putText(preview, f"Scanned: {scan_count} | Pending: {search_queue.qsize()} | Done: {len(results_list)}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if cooldown_counter > 0:
        auto_status = f"COOLDOWN: {cooldown_counter}"
        auto_color = (0, 165, 255)
    elif auto_capture_enabled:
        if peak_detected:
            auto_status = f"AUTO: Tracking... ({frames_since_peak}/{STABLE_FRAMES_REQUIRED})"
            auto_color = (0, 255, 255)
        else:
            auto_status = "AUTO: Ready"
            auto_color = (0, 255, 0)
    else:
        auto_status = "AUTO: OFF (A=toggle)"
        auto_color = (150, 150, 150)
    cv2.putText(preview, auto_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, auto_color, 2)

    cv2.putText(preview, "SPACE=Capture | R=Results | Q=Quit", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    focus_mode = "MANUAL" if manual_focus_mode else "AUTO"
    cv2.putText(preview, f"Focus: {focus_mode} | +/-=adjust", (10, preview.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Show preview
    preview_resized = cv2.resize(preview, (960, 540))
    cv2.imshow("Rapid Card Scanner", preview_resized)

    # Smart auto-capture logic
    should_capture = False
    if auto_capture_enabled and cooldown_counter == 0 and ids is not None and len(ids) == 4:
        if sharpness > best_sharpness and sharpness > AUTO_CAPTURE_THRESHOLD:
            best_sharpness = sharpness
            best_frame = frame.copy()
            best_markers = marker_bounding_boxes
            frames_since_peak = 0
            peak_detected = True
        elif peak_detected:
            frames_since_peak += 1

        if peak_detected and frames_since_peak >= STABLE_FRAMES_REQUIRED:
            if best_sharpness > AUTO_CAPTURE_THRESHOLD:
                frame = best_frame
                marker_bounding_boxes = best_markers
                should_capture = True
                # Reset
                best_sharpness = 0
                peak_detected = False
                frames_since_peak = 0
                cooldown_counter = COOLDOWN_FRAMES

    # Handle keys
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Manual capture
        if ids is not None and len(ids) == 4:
            should_capture = True
            cooldown_counter = COOLDOWN_FRAMES
            # Reset auto-capture state
            best_sharpness = 0
            peak_detected = False
            frames_since_peak = 0
        else:
            print(f"       Cannot capture - need 4 markers (found {len(ids) if ids is not None else 0})")
    elif key == ord('r'):  # Show results
        show_results = True
        break
    elif key == ord('q'):  # Quit
        break
    elif key == ord('a'):
        auto_capture_enabled = not auto_capture_enabled
        print(f"       Auto-capture: {'ON' if auto_capture_enabled else 'OFF'}")
    elif key == ord('f'):
        manual_focus_mode = not manual_focus_mode
        if manual_focus_mode:
            CAMERA.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            CAMERA.set(cv2.CAP_PROP_FOCUS, current_focus)
        else:
            CAMERA.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    elif key == ord('=') or key == ord('+'):
        if manual_focus_mode:
            current_focus = min(255, current_focus + 10)
            CAMERA.set(cv2.CAP_PROP_FOCUS, current_focus)
    elif key == ord('-'):
        if manual_focus_mode:
            current_focus = max(0, current_focus - 10)
            CAMERA.set(cv2.CAP_PROP_FOCUS, current_focus)
    elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
        new_dict_id = int(chr(key))
        if new_dict_id in ARUCO_DICTS:
            current_dict_id = new_dict_id
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[current_dict_id][1])
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Process capture
    if should_capture:
        scan_count += 1
        print(f"\n[Scan #{scan_count}] Capturing {MULTI_CAPTURE_COUNT} frames for consensus...")

        # Capture multiple frames for consensus voting
        card_images = []
        thumbnail = None

        for i in range(MULTI_CAPTURE_COUNT):
            # Use the best frame for first capture, then grab new frames
            if i == 0:
                current_frame = frame
                current_markers = marker_bounding_boxes
            else:
                # Small delay then grab a new frame
                time.sleep(0.05)  # 50ms delay between captures
                ret, current_frame = CAMERA.read()
                if not ret:
                    continue
                # Re-detect markers in the new frame
                corners, ids, _ = detector.detectMarkers(current_frame)
                if ids is None or len(ids) != 4:
                    continue
                current_markers = corners

            card_image = extract_card_from_frame(current_frame, current_markers)

            if card_image is not None:
                card_images.append(card_image)
                if thumbnail is None:
                    thumbnail = cv2.resize(card_image, (100, 140))

        if card_images:
            print(f"       Captured {len(card_images)} frames, queuing for search...")
            # Queue all images for consensus search
            search_queue.put((scan_count, card_images, thumbnail))
        else:
            print(f"       Failed to extract card from any frame")
            scan_count -= 1

# ============== CLEANUP ==============
cv2.destroyAllWindows()
CAMERA.release()

# Stop the worker
worker_running = False
search_queue.put(None)
search_thread.join(timeout=2)

# Wait for pending searches to complete
if search_queue.qsize() > 0 or len(results_list) < scan_count:
    print(f"\nWaiting for {scan_count - len(results_list)} pending searches...")
    timeout = 30
    start = time.time()
    while len(results_list) < scan_count and (time.time() - start) < timeout:
        time.sleep(0.5)

# ============== SHOW RESULTS ==============
if show_results and len(results_list) > 0:
    print("\n" + "=" * 60)
    print("SCAN RESULTS")
    print("=" * 60)

    # Sort by scan number
    results_list.sort(key=lambda x: x['scan_num'])

    for result in results_list:
        match = result['top_match']
        score = result['top_score']
        card_info = get_card_info(match['set'], match['num'])
        set_info = result.get('set_info')
        consensus_used = result.get('consensus_used', False)

        print(f"\nScan #{result['scan_num']}:")
        print(f"  {card_info['name']}")
        print(f"  Set: {card_info['set_name']} ({match['set']}) #{match['num']}")

        # Show confidence and matching info
        confidence_str = f"Confidence: {score:.1%}"
        if consensus_used:
            confidence_str += " [CONSENSUS]"
        if set_info:
            confidence_str += f" (Set symbol: {set_info['set_score']:.1%})"
        print(f"  Rarity: {card_info['rarity'].capitalize()} | {confidence_str}")

        if card_info['prices_usd']:
            print(f"  Price: ${card_info['prices_usd']}")

        # Show other matches
        all_results = result['all_results']
        if len(all_results['ids'][0]) > 1:
            print("  Other possibilities:")
            for i in range(1, min(3, len(all_results['ids'][0]))):
                meta = all_results['metadatas'][0][i]
                dist = all_results['distances'][0][i]
                alt_info = get_card_info(meta['set'], meta['num'])
                print(f"    - {alt_info['name']} ({meta['set']} #{meta['num']}) ({1-dist:.1%})")

    print("\n" + "=" * 60)
    print(f"Total cards scanned: {len(results_list)}")
    print("=" * 60)

    # Create visual summary
    if len(results_list) > 0:
        cols = min(5, len(results_list))
        rows = (len(results_list) + cols - 1) // cols
        cell_w, cell_h = 120, 180
        summary = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for i, result in enumerate(results_list):
            row = i // cols
            col = i % cols
            x = col * cell_w + 10
            y = row * cell_h + 10

            thumb = result['thumbnail']
            summary[y:y+thumb.shape[0], x:x+thumb.shape[1]] = thumb

            # Add label
            match = result['top_match']
            card_info = get_card_info(match['set'], match['num'])
            # Truncate name if too long
            name = card_info['name'][:14] + "..." if len(card_info['name']) > 17 else card_info['name']
            label = f"{match['set']}:{match['num']}"
            score_text = f"{result['top_score']:.0%}"
            cv2.putText(summary, name, (x, y + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(summary, f"{label} {score_text}", (x, y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        cv2.imshow("Scan Results (press any key to close)", summary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ============== EXPORT CSV ==============
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create exports folder if it doesn't exist
    exports_folder = "./exports"
    os.makedirs(exports_folder, exist_ok=True)

    csv_filename = os.path.join(exports_folder, f"scan_results_{timestamp}.csv")

    print(f"\nExporting results to {csv_filename}...")

    # Group cards by set:collector_number to count quantities
    card_counts = {}
    card_scores = {}  # Track best confidence score for each card

    for result in results_list:
        match = result['top_match']
        card_key = f"{match['set']}:{match['num']}"

        if card_key in card_counts:
            card_counts[card_key] += 1
            # Keep the best score
            if result['top_score'] > card_scores[card_key]:
                card_scores[card_key] = result['top_score']
        else:
            card_counts[card_key] = 1
            card_scores[card_key] = result['top_score']

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'Title',
            'Edition',
            'Foil',
            'Qty',
            'Set Code',
            'Collector Number',
            'Rarity',
            'Type',
            'Mana Cost',
            'Confidence',
            'Price (USD)',
            'Price (USD Foil)',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write unique cards with quantities
        for card_key, qty in card_counts.items():
            set_code, collector_num = card_key.split(':')
            card_info = get_card_info(set_code, collector_num)

            writer.writerow({
                'Title': card_info['name'],
                'Edition': card_info['set_name'],
                'Foil': 'false',
                'Qty': qty,
                'Set Code': card_info['set_code'],
                'Collector Number': card_info['collector_number'],
                'Rarity': card_info['rarity'],
                'Type': card_info['type_line'],
                'Mana Cost': card_info['mana_cost'],
                'Confidence': f"{card_scores[card_key]:.1%}",
                'Price (USD)': card_info['prices_usd'] or 'N/A',
                'Price (USD Foil)': card_info['prices_usd_foil'] or 'N/A',
            })

    unique_cards = len(card_counts)
    total_cards = sum(card_counts.values())
    print(f"Exported {unique_cards} unique cards ({total_cards} total) to {csv_filename}")

# ============== CLEANUP TEMP FILES ==============
print("\nCleaning up temporary files...")
temp_files = [
    "card_image.jpg",
    "card_image_original.jpg",
    "aruco_image.jpg",
    "pre_image.jpg",
    "edges.jpg",
    "debug_contour.jpg",
    "debug_frame.jpg",
    "contours.jpg",
    "approx_polygon.jpg",
    "largest_contour.jpg",
]

deleted_count = 0
for temp_file in temp_files:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            deleted_count += 1
        except Exception as e:
            print(f"       Warning: Could not delete {temp_file}: {e}")

if deleted_count > 0:
    print(f"       Deleted {deleted_count} temporary files")

print("\nDone!")
