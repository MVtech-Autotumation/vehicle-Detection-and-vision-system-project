from ultralytics import YOLO
import cv2 as c1
import math

model = YOLO("yolo11s.pt")

video_path = r"D:\MVtech basic train model\traffic model1.1\traffic_SampleVideo_LowQuality.mp4"

webcam = c1.VideoCapture(video_path)

if not webcam.isOpened():
    print("ERROR: Could not open video.")
    print("Check the video path.")
    exit()


fps = webcam.get(c1.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

print("Video FPS:", fps)

frame_time = 1 / fps



class_list = model.names

# Only keep vehicle classes
vehicle_classes = {
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle"
}

# ------------------------------------------------
# SPEED CALIBRATION
# ------------------------------------------------

REAL_DISTANCE_METERS = 20.0
IMAGE_DISTANCE_PIXELS = 400.0

PIXEL_TO_METER = REAL_DISTANCE_METERS / IMAGE_DISTANCE_PIXELS

# ------------------------------------------------
# STORE VEHICLE DATA
# ------------------------------------------------

vehicle_data = {}

# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------

while True:

    # Read frame
    ret, frame = webcam.read()

    if not ret:
        print("Video ended or frame could not be read.")
        break

    # ------------------------------------------------
    # YOLO DETECTION + TRACKING
    # ------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:

        # Get detection data
        coordinates = boxes.xyxy.cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        # ------------------------------------------------
        # PROCESS EACH VEHICLE
        # ------------------------------------------------

        for box, track_id, class_id, confidence in zip(
            coordinates,
            track_ids,
            class_ids,
            confidences
        ):

            # Bounding box
            x1, y1, x2, y2 = map(int, box)

            # Class name
            class_name = class_list[class_id]

            # ------------------------------------------------
            # IGNORE NON-VEHICLES
            # ------------------------------------------------

            if class_name not in vehicle_classes:
                continue

            # ------------------------------------------------
            # BOTTOM CENTER OF VEHICLE
            # ------------------------------------------------

            center_x = int((x1 + x2) / 2)
            bottom_y = int(y2)

            current_position = (center_x, bottom_y)

            # Default speed
            speed_kmph = 0.0

            # ------------------------------------------------
            # SPEED CALCULATION
            # ------------------------------------------------

            if track_id in vehicle_data:

                previous_position = vehicle_data[track_id]["position"]

                # Pixel movement
                dx = current_position[0] - previous_position[0]
                dy = current_position[1] - previous_position[1]

                pixel_distance = math.sqrt(
                    dx ** 2 + dy ** 2
                )

                # Pixel speed
                pixel_speed = pixel_distance / frame_time

                # Convert pixel speed to m/s
                meter_speed = pixel_speed * PIXEL_TO_METER

                # Convert m/s to km/h
                speed_kmph = meter_speed * 3.6

            # ------------------------------------------------
            # SAVE CURRENT POSITION
            # ------------------------------------------------

            vehicle_data[track_id] = {
                "position": current_position
            }

            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            c1.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # DRAW TRACKING POINT
            # ------------------------------------------------

            c1.circle(
                frame,
                current_position,
                6,
                (0, 0, 255),
                -1
            )

            # ------------------------------------------------
            # DISPLAY INFORMATION
            # ------------------------------------------------

            text = (
                f"ID: {track_id} | "
                f"{class_name} | "
                f"{confidence:.2f} | "
                f"{speed_kmph:.1f} km/h"
            )

            c1.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                c1.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

    # ------------------------------------------------
    # DISPLAY
    # ------------------------------------------------

    c1.imshow(
        "Vehicle Speed Tracking",
        frame
    )

    # Press Q to quit
    key = c1.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ------------------------------------------------
# CLOSE
# ------------------------------------------------

webcam.release()
c1.destroyAllWindows()