import cv2
import numpy as np
from itertools import combinations

print("=" * 50)
print("MTG Card Capture")
print("=" * 50)

#Open out webcam
print("\n[1/5] Opening webcam...")

# Try to find an available camera
CAMERA = None
camera_index = None

for idx in range(5):  # Try camera indices 0-4
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
    print("\nTroubleshooting tips:")
    print("  1. Make sure your webcam is plugged in")
    print("  2. Close any other apps using the camera (Zoom, Teams, etc.)")
    print("  3. Try unplugging and replugging the webcam")
    print("  4. Check Device Manager for camera issues")
    print("  5. Restart your computer if nothing else works")
    exit(1)

CAMERA.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
CAMERA.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Try to enable autofocus first
CAMERA.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# If autofocus doesn't work well, we'll allow manual adjustment
current_focus = 0  # Start with autofocus (0 = auto on some cameras)
manual_focus_mode = False

print(f"       Webcam opened successfully! (index {camera_index})")
print("       Focus controls: F=Toggle auto/manual, +/- to adjust manual focus")

# Live preview mode - press SPACE to capture, Q to quit
print("\n[2/5] Live preview mode")
print("       - Press SPACE to capture when card is positioned")
print("       - Press Q to quit")
print("       - Press D to save debug image")
print("       - Press 1-5 to try different ArUco dictionaries")
print("       - Make sure all 4 ArUco markers are visible")

# Try multiple ArUco dictionaries - your markers might be from a different set
ARUCO_DICTS = {
    1: ("4X4_50", cv2.aruco.DICT_4X4_50),
    2: ("4X4_100", cv2.aruco.DICT_4X4_100),
    3: ("5X5_50", cv2.aruco.DICT_5X5_50),
    4: ("6X6_50", cv2.aruco.DICT_6X6_50),
    5: ("ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
}

current_dict_id = 4  # 6X6_50 - works best for your markers
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[current_dict_id][1])
parameters = cv2.aruco.DetectorParameters()

# Adjust parameters for better detection
parameters.adaptiveThreshConstant = 7
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.minMarkerPerimeterRate = 0.03  # Allow smaller markers
parameters.maxMarkerPerimeterRate = 4.0
parameters.polygonalApproxAccuracyRate = 0.03
parameters.minCornerDistanceRate = 0.05
parameters.minDistanceToBorder = 3

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
print(f"       Using ArUco dictionary: {ARUCO_DICTS[current_dict_id][0]}")

while True:
    returncode, frame = CAMERA.read()
    if not returncode:
        print("ERROR: Camera read failed")
        continue

    # Detect markers for preview
    marker_bounding_boxes, ids, rejected = detector.detectMarkers(frame)

    # Draw detected markers on preview
    preview = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(preview, marker_bounding_boxes, ids)
        status = f"Markers found: {len(ids)}/4 (Dict: {ARUCO_DICTS[current_dict_id][0]})"
        color = (0, 255, 0) if len(ids) == 4 else (0, 165, 255)
        # Show which marker IDs were found
        id_list = [str(i[0]) for i in ids]
        cv2.putText(preview, f"IDs: {', '.join(id_list)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        status = f"No markers found (Dict: {ARUCO_DICTS[current_dict_id][0]})"
        color = (0, 0, 255)

    # Show rejected candidates count (helps with debugging)
    if rejected is not None and len(rejected) > 0:
        cv2.putText(preview, f"Rejected candidates: {len(rejected)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

    # Add status text to preview
    cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(preview, "SPACE=Capture, Q=Quit, D=Debug, 1-5=Dict", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show focus info
    focus_mode = "MANUAL" if manual_focus_mode else "AUTO"
    cv2.putText(preview, f"Focus: {focus_mode} (F=toggle, +/-=adjust) Value: {current_focus}", (10, preview.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show smaller preview window
    preview_resized = cv2.resize(preview, (960, 540))
    cv2.imshow("Card Capture Preview", preview_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # SPACE to capture
        if ids is not None and len(ids) == 4:
            print("       Captured! Processing...")
            break
        else:
            print(f"       Cannot capture - only {len(ids) if ids is not None else 0} markers visible")
    elif key == ord('q'):  # Q to quit
        print("       Cancelled by user")
        CAMERA.release()
        cv2.destroyAllWindows()
        exit(0)
    elif key == ord('d'):  # D to save debug image
        debug_path = "debug_frame.jpg"
        cv2.imwrite(debug_path, frame)
        print(f"       Saved debug image to {debug_path}")
    elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
        # Switch ArUco dictionary
        new_dict_id = int(chr(key))
        if new_dict_id in ARUCO_DICTS:
            current_dict_id = new_dict_id
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[current_dict_id][1])
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            print(f"       Switched to ArUco dictionary: {ARUCO_DICTS[current_dict_id][0]}")
    elif key == ord('f'):  # Toggle focus mode
        manual_focus_mode = not manual_focus_mode
        if manual_focus_mode:
            CAMERA.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
            CAMERA.set(cv2.CAP_PROP_FOCUS, current_focus)
            print(f"       Switched to MANUAL focus (value: {current_focus})")
        else:
            CAMERA.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            print("       Switched to AUTO focus")
    elif key == ord('=') or key == ord('+'):  # Increase focus
        if manual_focus_mode:
            current_focus = min(255, current_focus + 10)
            CAMERA.set(cv2.CAP_PROP_FOCUS, current_focus)
            print(f"       Focus: {current_focus}")
    elif key == ord('-'):  # Decrease focus
        if manual_focus_mode:
            current_focus = max(0, current_focus - 10)
            CAMERA.set(cv2.CAP_PROP_FOCUS, current_focus)
            print(f"       Focus: {current_focus}")

cv2.destroyAllWindows()

#Set up input and outputs
pre_image = frame.copy()
cv2.imwrite("pre_image.jpg", pre_image)
print("       Saved raw image to pre_image.jpg")

aruco_cropped_image = None

#Use OpenCV to find the markers (already detected above)
print("\n[3/5] Processing ArUco markers...")

# Make sure we found our 4 markers
if ids is not None and len(ids) == 4:
    print(f"       Found all 4 markers!")

    #We want to...
    # Find the markers
    # Get their location
    # And crop the image down to the area inside the markers, without including any of the marker
    centers = []
    max_width = 0
    max_height = 0

    for box in marker_bounding_boxes:
        box_croners = box[0]
        center_x = int(box_croners[:, 0].mean()) #Sum all the X coordinates and get the average for the center X
        center_y = int(box_croners[:, 1].mean()) # Do the same for the Y

        width = int(box_croners[:, 0].max() - box_croners[:, 0].min()) #Find the two points most opposite on the X axis, and subtract them to get the width
        height = int(box_croners[:, 1].max() - box_croners[:, 1].min()) #Do the same for the Y

        if width > max_width:
            max_width = width #Keep track of the widest marker's width
        if height > max_height:
            max_height = height #Keep track of the tallest marker's height

        centers.append((center_x, center_y))

    # We'll now basically find the x and y of the markers, and then offset by the max width and height so we are inside the markers
    min_x = min(centers, key=lambda x: x[0])[0] + max_width / 2
    max_x = max(centers, key=lambda x: x[0])[0] - max_width / 2
    min_y = min(centers, key=lambda x: x[1])[1] + max_height / 2
    max_y = max(centers, key=lambda x: x[1])[1] - max_height / 2

    # Defind the four inside corners of the markers
    centers_np = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ], dtype="float32")

    #Create a rectangle from the corners
    rect = cv2.convexHull(centers_np)
    rect = np.array(rect, dtype="float32")

    # Define the dimensions of the cropped area (desired output size)
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    # This allows us to make the image "square" and auto rotate it to be straight
    M = cv2.getPerspectiveTransform(rect, dst)

    # Perform the warp
    warped = cv2.warpPerspective(pre_image, M, (width, height))
    aruco_cropped_image = warped.copy()
    print("       Perspective correction complete!")
else:
    print("ERROR: Could not find all 4 ArUco markers")
    CAMERA.release()
    exit(1)

cv2.imwrite("aruco_image.jpg", aruco_cropped_image)
print("       Saved cropped area to aruco_image.jpg")

print("\n[4/5] Detecting card edges...")
#By now we have just the inside of our markers, it time to find the card and essentially do the same thing

# Convert to grayscale
gray = cv2.cvtColor(aruco_cropped_image, cv2.COLOR_BGR2GRAY)

# Try multiple edge detection approaches and pick the best one
best_contours = []
best_method = ""

# Method 1: Adaptive thresholding (works better with varying lighting)
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
blurred1 = cv2.GaussianBlur(adaptive_thresh, (5, 5), 0)
edges1 = cv2.Canny(blurred1, 30, 100)
contours1, _ = cv2.findContours(edges1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
if len(contours1) > len(best_contours):
    best_contours = contours1
    best_method = "adaptive"
    edges = edges1

# Method 2: Direct Canny on grayscale (works well with good contrast)
blurred2 = cv2.GaussianBlur(gray, (5, 5), 0)
edges2 = cv2.Canny(blurred2, 50, 150)
contours2, _ = cv2.findContours(edges2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
if len(contours2) > len(best_contours):
    best_contours = contours2
    best_method = "direct_canny"
    edges = edges2

# Method 3: Otsu's thresholding (automatically finds best threshold)
blurred3 = cv2.GaussianBlur(gray, (5, 5), 0)
ret, otsu_thresh = cv2.threshold(blurred3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges3 = cv2.Canny(otsu_thresh, 50, 150)
contours3, _ = cv2.findContours(edges3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
if len(contours3) > len(best_contours):
    best_contours = contours3
    best_method = "otsu"
    edges = edges3

# Method 4: Color-based edge detection (detects card against any background)
# Convert to LAB color space for better edge detection
lab = cv2.cvtColor(aruco_cropped_image, cv2.COLOR_BGR2LAB)
l_channel = lab[:,:,0]
blurred4 = cv2.GaussianBlur(l_channel, (5, 5), 0)
edges4 = cv2.Canny(blurred4, 30, 100)
contours4, _ = cv2.findContours(edges4, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
if len(contours4) > len(best_contours):
    best_contours = contours4
    best_method = "lab_color"
    edges = edges4

contours = best_contours
cv2.imwrite("edges.jpg", edges)

if len(contours) == 0:
    print("ERROR: No contours found - make sure card is visible and has contrast with background")
    CAMERA.release()
    exit(1)

print(f"       Found {len(contours)} contours using {best_method} method, analyzing...")

# MTG cards have an aspect ratio of approximately 63mm x 88mm = ~0.716 (width/height)
# We'll look for contours that are:
# 1. Large enough (at least 10% of the image area)
# 2. Roughly rectangular (4 corners when approximated)
# 3. Close to the expected aspect ratio

image_area = aruco_cropped_image.shape[0] * aruco_cropped_image.shape[1]
min_card_area = image_area * 0.05  # Card should be at least 5% of cropped area

# Sort contours by area (largest first)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

best_contour = None
for contour in sorted_contours[:10]:  # Check top 10 largest contours
    area = cv2.contourArea(contour)
    if area < min_card_area:
        continue

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx_poly = cv2.approxPolyDP(contour, epsilon, True)

    # Check if it's roughly rectangular (4-6 vertices)
    if 4 <= len(approx_poly) <= 6:
        # Get bounding rectangle to check aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = min(w, h) / max(w, h)

        # MTG card aspect ratio is ~0.716, allow some tolerance (0.5 to 0.9)
        if 0.5 <= aspect_ratio <= 0.9:
            best_contour = contour
            print(f"       Found card-like contour: {w}x{h} pixels, aspect ratio: {aspect_ratio:.3f}")
            break

# Fallback to largest contour if no card-shaped contour found
if best_contour is None:
    print("       Warning: No card-shaped contour found, using largest contour")
    best_contour = sorted_contours[0]

largest_contour = best_contour
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# Save debug image showing detected contour
debug_contour_img = aruco_cropped_image.copy()
cv2.drawContours(debug_contour_img, [largest_contour], -1, (0, 255, 0), 3)
cv2.drawContours(debug_contour_img, [approx], -1, (0, 0, 255), 2)
cv2.imwrite("debug_contour.jpg", debug_contour_img)
print(f"       Saved contour debug image to debug_contour.jpg")

# We might have something that isn't a perfect rectangle
# So we'll try to pick points along our shape and pick the 4 that make the largest area
# This should be the 4 corners of the card
max_area = 0
best_quad = None

# If we have fewer than 4 points, use the bounding rectangle instead
if len(approx) < 4:
    print("       Warning: Contour has fewer than 4 points, using bounding rectangle")
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    best_quad = np.array(box, dtype="float32").reshape(4, 1, 2)
else:
    # We'll try every combination of 4 points and pick the one with the largest area
    for quad in combinations(approx, 4):
        quad = np.array(quad, dtype="float32")
        area = cv2.contourArea(quad)
        if area > max_area:
            max_area = area
            best_quad = quad

if best_quad is None:
    print("ERROR: Could not find 4 corner points for the card")
    print("       Try adjusting the card position or lighting")
    CAMERA.release()
    exit(1)

points = np.squeeze(best_quad) #Best quad is a list of a list of points, we just want a list of points

# Make sure points is the right shape
if points.ndim == 1:
    points = points.reshape(4, 2)

#Sort the points so their in clockwise order ie
# . Origin
#
#   0 1
#   3 2
# 
# v+Y   >+X
rect = np.zeros((4, 2), dtype="float32")
s = points.sum(axis=1) #Add together each points X and Y
rect[0] = points[np.argmin(s)] #Top left will have the smallest X, and smallest Y, and as such the smallest sum
rect[2] = points[np.argmax(s)] #Bottom right will have the largest X, and largest Y, and as such the largest sum
diff = np.diff(points, axis=1) #Subtract the X from the Y
rect[1] = points[np.argmin(diff)] #Top right will have the largest X and largest Y, so the smallest difference
rect[3] = points[np.argmax(diff)] #Bottom left will have the smallest X and smallest Y, so the largest difference


top_width =  rect[1][0] - rect[0][0] 
bottom_width = rect[3][0] - rect[2][0]

left_height = rect[0][1] - rect[3][1]
right_height = rect[2][1] - rect[1][1]

width = int(abs(max(top_width, bottom_width)))
height = int(abs(max(left_height, right_height)))

# Sanity check - make sure dimensions are reasonable
if width < 50 or height < 50:
    print(f"ERROR: Detected card too small ({width}x{height}). Check card position.")
    CAMERA.release()
    exit(1)

print(f"       Card detected! Size: {width}x{height} pixels")

# Define the destination points
dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")

# Perform the perspective transformation
matrix = cv2.getPerspectiveTransform(rect, dst)
card_image = cv2.warpPerspective(aruco_cropped_image, matrix, (width, height))

print("\n[5/5] Saving card image...")
cv2.imwrite("card_image.jpg", card_image)
print("       Saved card to card_image.jpg")

# Release camera
CAMERA.release()

# Show the final result
print("\n" + "=" * 50)
print("SUCCESS! Card captured successfully!")
print("=" * 50)
print(f"\nOutput files:")
print(f"  - pre_image.jpg     (raw camera image)")
print(f"  - aruco_image.jpg   (cropped to markers)")
print(f"  - edges.jpg         (edge detection)")
print(f"  - card_image.jpg    (final card - use this for search)")
print(f"\nNext step: Run 'python search.py' to identify the card")

# Show final card image
cv2.imshow("Captured Card (press any key to close)", card_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
