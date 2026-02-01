import cv2
import numpy as np
from itertools import combinations
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import chromadb

print("=" * 50)
print("MTG Card Capture & Search")
print("=" * 50)

# Load the VGG16 model early (takes a moment)
print("\n[0/6] Loading AI model...")
nn = VGG16(weights='imagenet', include_top=False)
print("       Model loaded!")

# Connect to ChromaDB
CHROMA_DB_PATH = "./chroma_db"
print("       Connecting to card database...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name="mtg_cards")
print(f"       Database connected! ({collection.count():,} cards)")

# ============== CAMERA SETUP ==============
print("\n[1/6] Opening webcam...")

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

print(f"       Webcam opened successfully! (index {camera_index})")

# ============== IMAGE QUALITY FUNCTIONS ==============
def calculate_sharpness(image):
    """Calculate image sharpness using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def calculate_contrast(image):
    """Calculate image contrast using standard deviation"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

def enhance_card_image(image):
    """Enhance card image for better matching"""
    # 1. Convert to LAB color space for better processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # 3. Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 4. Slight sharpening
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # 5. Blend original with sharpened (50% each for subtle effect)
    result = cv2.addWeighted(enhanced, 0.5, sharpened, 0.5, 0)

    return result

# Track quality metrics for smart auto-capture
sharpness_history = []
MAX_HISTORY = 60  # Track more frames for better peak detection
AUTO_CAPTURE_THRESHOLD = 80  # Minimum sharpness for auto-capture
STABLE_FRAMES_REQUIRED = 10  # Frames of stable/declining sharpness before capture
auto_capture_enabled = True  # Enable by default now
peak_detected = False
frames_since_peak = 0
best_frame = None
best_sharpness = 0

# ============== ARUCO SETUP ==============
print("\n[2/6] Live preview mode")
print("       Controls:")
print("         SPACE  = Capture manually")
print("         A      = Toggle auto-capture (captures when sharp)")
print("         F      = Toggle auto/manual focus")
print("         +/-    = Adjust manual focus")
print("         1-5    = Switch ArUco dictionary")
print("         Q      = Quit")

ARUCO_DICTS = {
    1: ("4X4_50", cv2.aruco.DICT_4X4_50),
    2: ("4X4_100", cv2.aruco.DICT_4X4_100),
    3: ("5X5_50", cv2.aruco.DICT_5X5_50),
    4: ("6X6_50", cv2.aruco.DICT_6X6_50),
    5: ("ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
}

current_dict_id = 4  # 6X6_50
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

# ============== MAIN CAPTURE LOOP ==============
while True:
    returncode, frame = CAMERA.read()
    if not returncode:
        continue

    # Calculate sharpness
    sharpness = calculate_sharpness(frame)
    sharpness_history.append(sharpness)
    if len(sharpness_history) > MAX_HISTORY:
        sharpness_history.pop(0)

    avg_sharpness = sum(sharpness_history) / len(sharpness_history) if sharpness_history else 0
    max_sharpness = max(sharpness_history) if sharpness_history else 0

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
    sharpness_normalized = min(sharpness / 200, 1.0)  # Normalize to 0-1
    bar_width = int(300 * sharpness_normalized)
    bar_color = (0, 255, 0) if sharpness > AUTO_CAPTURE_THRESHOLD else (0, 165, 255)
    cv2.rectangle(preview, (10, 180), (310, 210), (50, 50, 50), -1)
    cv2.rectangle(preview, (10, 180), (10 + bar_width, 210), bar_color, -1)
    cv2.putText(preview, f"Sharpness: {int(sharpness)}", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Status text
    cv2.putText(preview, marker_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, marker_color, 2)

    if auto_capture_enabled:
        if peak_detected:
            auto_status = f"AUTO: Tracking peak... ({frames_since_peak}/{STABLE_FRAMES_REQUIRED})"
            auto_color = (0, 255, 255)  # Yellow - tracking
        else:
            auto_status = "AUTO: Waiting for focus..."
            auto_color = (255, 255, 0)  # Cyan - waiting
    else:
        auto_status = "AUTO-CAPTURE: OFF (A=on)"
        auto_color = (150, 150, 150)
    cv2.putText(preview, auto_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, auto_color, 2)

    cv2.putText(preview, "SPACE=Capture, Q=Quit, F=Focus", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    focus_mode = "MANUAL" if manual_focus_mode else "AUTO"
    cv2.putText(preview, f"Focus: {focus_mode} (+/-) Val:{current_focus}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show preview
    preview_resized = cv2.resize(preview, (960, 540))
    cv2.imshow("Card Capture & Search", preview_resized)

    # Smart auto-capture logic - waits for peak sharpness
    should_capture = False
    if auto_capture_enabled and ids is not None and len(ids) == 4:
        # Track the best frame we've seen
        if sharpness > best_sharpness and sharpness > AUTO_CAPTURE_THRESHOLD:
            best_sharpness = sharpness
            best_frame = frame.copy()
            frames_since_peak = 0
            peak_detected = True
        elif peak_detected:
            frames_since_peak += 1

        # If we've seen a peak and sharpness has been stable/declining for a while, capture
        if peak_detected and frames_since_peak >= STABLE_FRAMES_REQUIRED:
            if best_sharpness > AUTO_CAPTURE_THRESHOLD:
                frame = best_frame  # Use the best frame we captured
                should_capture = True
                print(f"       Auto-captured at peak sharpness: {int(best_sharpness)}")
                # Reset for next capture
                best_sharpness = 0
                peak_detected = False
                frames_since_peak = 0

    # Handle key input
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Manual capture
        if ids is not None and len(ids) == 4:
            should_capture = True
            print(f"       Manual capture! Sharpness: {int(sharpness)}")
        else:
            print(f"       Cannot capture - need 4 markers (found {len(ids) if ids is not None else 0})")
    elif key == ord('q'):
        print("       Cancelled by user")
        CAMERA.release()
        cv2.destroyAllWindows()
        exit(0)
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

    if should_capture:
        break

cv2.destroyAllWindows()

# ============== PROCESS ARUCO MARKERS ==============
print("\n[3/6] Processing ArUco markers...")
pre_image = frame.copy()

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
aruco_cropped_image = cv2.warpPerspective(pre_image, M, (width, height))
print("       Perspective correction complete!")

# ============== DETECT CARD EDGES ==============
print("\n[4/6] Detecting card edges...")

gray = cv2.cvtColor(aruco_cropped_image, cv2.COLOR_BGR2GRAY)

# Try multiple edge detection methods
best_contours = []
edges = None

for method_name, method_func in [
    ("adaptive", lambda g: cv2.Canny(cv2.GaussianBlur(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2), (5,5), 0), 30, 100)),
    ("direct_canny", lambda g: cv2.Canny(cv2.GaussianBlur(g, (5,5), 0), 50, 150)),
    ("otsu", lambda g: cv2.Canny(cv2.threshold(cv2.GaussianBlur(g, (5,5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 50, 150)),
]:
    e = method_func(gray)
    c, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(c) > len(best_contours):
        best_contours = c
        edges = e

contours = best_contours

if len(contours) == 0:
    print("ERROR: No contours found")
    CAMERA.release()
    exit(1)

# Find card-shaped contour
image_area = aruco_cropped_image.shape[0] * aruco_cropped_image.shape[1]
min_card_area = image_area * 0.05
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

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
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    best_quad = np.array(box, dtype="float32").reshape(4, 1, 2)
else:
    for quad in combinations(approx, 4):
        quad = np.array(quad, dtype="float32")
        area = cv2.contourArea(quad)
        if area > max_area:
            max_area = area
            best_quad = quad

if best_quad is None:
    print("ERROR: Could not find card corners")
    CAMERA.release()
    exit(1)

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
    print(f"ERROR: Card too small ({width}x{height})")
    CAMERA.release()
    exit(1)

print(f"       Card detected: {width}x{height} pixels")

dst = np.array([
    [0, 0], [width - 1, 0],
    [width - 1, height - 1], [0, height - 1]
], dtype="float32")

matrix = cv2.getPerspectiveTransform(rect, dst)
card_image = cv2.warpPerspective(aruco_cropped_image, matrix, (width, height))

# Save original card image
cv2.imwrite("card_image_original.jpg", card_image)

# Enhance card image for better matching
print("       Enhancing image contrast and sharpness...")
card_image = enhance_card_image(card_image)
cv2.imwrite("card_image.jpg", card_image)
print("       Saved enhanced card to card_image.jpg")

# ============== SEARCH DATABASE ==============
print("\n[5/6] Searching database...")

# Prepare image for VGG16
img_resized = cv2.resize(card_image, (224, 224))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_array = np.expand_dims(img_rgb, axis=0).astype('float32')
x = preprocess_input(img_array)

# Get embedding
preds = nn.predict(x, verbose=None)
vector = preds.flatten().tolist()

# Query ChromaDB
results = collection.query(
    query_embeddings=[vector],
    n_results=5
)

# ============== DISPLAY RESULTS ==============
print("\n[6/6] Results:")
print("=" * 50)

for i in range(len(results['ids'][0])):
    card_id = results['ids'][0][i]
    metadata = results['metadatas'][0][i]
    distance = results['distances'][0][i]
    score = 1 - distance

    rank_marker = ">>>" if i == 0 else "   "
    print(f"{rank_marker} {i+1}. Set: {metadata['set']}, Card #: {metadata['num']} (Score: {score:.3f})")

print("=" * 50)

# Show the captured card
print("\nPress any key to close...")
cv2.imshow("Captured Card", card_image)
cv2.waitKey(0)

# Cleanup
CAMERA.release()
cv2.destroyAllWindows()

print("\nDone!")
