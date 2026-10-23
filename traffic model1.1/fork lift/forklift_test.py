from ultralytics import YOLO
import cv2 as cv

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"D:\MVtech basic train model\traffic model1.1\best.pt"
VIDEO_PATH = r"D:\MVtech basic train model\traffic model1.1\forklift_video2.0.mp4"

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)

# ============================================================
# OPEN VIDEO
# ============================================================

video = cv.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("ERROR: Could not open video.")
    exit()

# ============================================================
# TRACKING
# ============================================================

while True:

    ret, frame = video.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.30
    )

    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:

        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()

        for box, track_id, class_id, confidence in zip(
            coordinates,
            track_ids,
            class_ids,
            confidences
        ):

            x1, y1, x2, y2 = map(int, box)

            class_name = model.names[class_id]

            text = (
                f"ID: {track_id} | "
                f"{class_name} | "
                f"{confidence:.2f}"
            )

            cv.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    cv.imshow("Forklift Tracking", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv.destroyAllWindows()