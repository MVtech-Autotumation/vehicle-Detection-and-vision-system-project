from ultralytics import YOLO
import cv2
import math

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11s.pt"

# Camera / video
VIDEO_SOURCE = 0
# For Hikvision use:
# VIDEO_SOURCE = "rtsp://username:password@192.168.0.21:554/Streaming/channels/102"

# ============================================================
# OBJECT REAL HEIGHTS
# ============================================================
# Approximate real-world heights in meters.
# These are only initial values for testing.

OBJECT_HEIGHTS = {
    "person": 1.70,
    "bottle": 0.25,
    "chair": 0.90,
    "backpack": 0.45,
    "laptop": 0.02,
    "cup": 0.12,
    "cell phone": 0.15,
    "suitcase": 0.70,
}

# ============================================================
# CAMERA CALIBRATION
# ============================================================
# You MUST calibrate this value for your camera.
#
# Example:
# Put a bottle at exactly 1 meter from camera.
# Measure its pixel height.
#
# FOCAL_LENGTH = pixel_height * distance / real_height

FOCAL_LENGTH = 700


# ============================================================
# LOAD YOLO
# ============================================================

model = YOLO(MODEL_PATH)

# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("ERROR: Could not open camera/video")
    exit()

print("Camera connected successfully!")


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def calculate_distance(real_height, pixel_height, focal_length):

    if pixel_height <= 0:
        return None

    distance = (real_height * focal_length) / pixel_height

    return distance


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame")
        break

    # YOLO detection
    results = model(frame, conf=0.40, verbose=False)

    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Class ID
            class_id = int(box.cls[0])

            # Class name
            class_name = model.names[class_id]

            # Only calculate distance for objects
            # that we have a known approximate height for.
            if class_name not in OBJECT_HEIGHTS:
                continue

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confidence
            confidence = float(box.conf[0])

            # Object pixel height
            pixel_height = y2 - y1

            # Real-world approximate height
            real_height = OBJECT_HEIGHTS[class_name]

            # Calculate distance
            distance = calculate_distance(
                real_height,
                pixel_height,
                FOCAL_LENGTH
            )

            if distance is None:
                continue

            # ==================================================
            # DRAW BOUNDING BOX
            # ==================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ==================================================
            # DISPLAY INFORMATION
            # ==================================================

            label = (
                f"{class_name} "
                f"{confidence:.2f} "
                f"{distance:.2f} m"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ==================================================
            # BOTTOM CENTER POINT
            # ==================================================

            center_x = int((x1 + x2) / 2)
            bottom_y = y2

            cv2.circle(
                frame,
                (center_x, bottom_y),
                5,
                (0, 0, 255),
                -1
            )

    # ============================================================
    # DISPLAY
    # ============================================================

    cv2.imshow("YOLO Object Distance", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()