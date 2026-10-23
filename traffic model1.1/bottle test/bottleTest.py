from ultralytics import YOLO
import cv2

# Settings
MODEL_PATH = "yolo11s.pt"
VIDEO_PATH = r" D:\MVtech basic train model\traffic model1.1\bottle test\bottle test video.mp4"

CAMERA_HEIGHT_M = 0.1245
FY = 700.0
CY = 512.0

# Load YOLO
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


def calculate_distance(bottom_y):
    difference = bottom_y - CY 

    if difference <= 0:
        return None

    return (FY * CAMERA_HEIGHT_M) / difference


# Main loop
while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        conf=0.25,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            # Detect only bottles
            if class_name != "bottle":
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Bottom-center of bottle
            bottom_x = int((x1 + x2) / 2)
            bottom_y = y2

            # Calculate distance
            distance = calculate_distance(bottom_y)

            if distance is None:
                continue

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw bottom point
            cv2.circle(
                frame,
                (bottom_x, bottom_y),
                6,
                (0, 0, 255),
                -1
            )

            # Distance text
            text = f"Distance: {distance:.2f} m"

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            # Confidence
            cv2.putText(
                frame,
                f"Bottle: {confidence:.2f}",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # Display
    cv2.imshow(
        "Bottle Distance Measurement",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
cap.release()
cv2.destroyAllWindows()