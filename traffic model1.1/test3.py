from ultralytics import YOLO
import cv2
import math
import threading
import time
from collections import defaultdict, deque


# ============================================================
# RTSP CAMERA CLASS
# ============================================================

class RTSPCamera:

    def __init__(self, rtsp_url):

        self.rtsp_url = rtsp_url

        self.cap = cv2.VideoCapture(
            rtsp_url,
            cv2.CAP_FFMPEG
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.frame = None
        self.running = True

        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.thread.start()


    def update(self):

        while self.running:

            ret, frame = self.cap.read()

            if ret:

                # Keep only the newest frame
                with self.lock:

                    self.frame = frame

            else:

                print("Frame lost. Reconnecting...")

                self.cap.release()

                time.sleep(1)

                self.cap = cv2.VideoCapture(
                    self.rtsp_url,
                    cv2.CAP_FFMPEG
                )

                self.cap.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1
                )


    def read(self):

        with self.lock:

            if self.frame is None:

                return False, None

            return True, self.frame.copy()


    def stop(self):

        self.running = False

        self.cap.release()


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = YOLO("yolo11s.pt")


# ============================================================
# HIKVISION RTSP
# ============================================================

# Use your NEW password here.
# If the password contains @, encode it as %40.

rtsp_url = (
    "rtsp://admin:shree@123@"
    "192.168.0.21:554/"
    "Streaming/channels/101"
)


# ============================================================
# START CAMERA
# ============================================================

camera = RTSPCamera(rtsp_url)

print("Starting Hikvision camera...")


# ============================================================
# CLASS NAMES
# ============================================================

class_list = model.names


# ============================================================
# VEHICLE CLASSES
# ============================================================

vehicle_classes = {
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle"
}


# ============================================================
# SPEED CALIBRATION
# ============================================================

# THESE ARE ONLY EXAMPLE VALUES.
# We will replace these with actual factory measurements later.

REAL_DISTANCE_METERS = 20.0
IMAGE_DISTANCE_PIXELS = 400.0

PIXEL_TO_METER = (
    REAL_DISTANCE_METERS /
    IMAGE_DISTANCE_PIXELS
)


# ============================================================
# VEHICLE TRACKING DATA
# ============================================================

# Previous position of each vehicle
vehicle_positions = {}


# Speed history for smoothing
speed_history = defaultdict(
    lambda: deque(maxlen=10)
)


# Previous timestamp of each vehicle
vehicle_times = {}


# ============================================================
# MAIN LOOP
# ============================================================

while True:


    # --------------------------------------------------------
    # GET LATEST CAMERA FRAME
    # --------------------------------------------------------

    ret, frame = camera.read()


    if not ret:

        time.sleep(0.01)

        continue


    # Current real time
    current_time = time.perf_counter()


    # --------------------------------------------------------
    # YOLO + BYTE TRACK
    # --------------------------------------------------------

    results = model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=0.25,

        verbose=False
    )


    boxes = results[0].boxes


    # --------------------------------------------------------
    # CHECK TRACKING IDS
    # --------------------------------------------------------

    if boxes is not None and boxes.id is not None:


        coordinates = boxes.xyxy.cpu()

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )

        class_ids = (
            boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            boxes.conf
            .cpu()
            .tolist()
        )


        # ----------------------------------------------------
        # PROCESS EACH OBJECT
        # ----------------------------------------------------

        for (
            box,
            track_id,
            class_id,
            confidence
        ) in zip(

            coordinates,
            track_ids,
            class_ids,
            confidences

        ):


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # CLASS NAME
            # ------------------------------------------------

            class_name = class_list[class_id]


            # ------------------------------------------------
            # IGNORE NON-VEHICLES
            # ------------------------------------------------

            if class_name not in vehicle_classes:

                continue


            # ------------------------------------------------
            # BOTTOM CENTER
            # ------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            bottom_y = int(y2)


            current_position = (
                center_x,
                bottom_y
            )


            # ------------------------------------------------
            # DEFAULT SPEED
            # ------------------------------------------------

            speed_kmph = 0.0


            # ------------------------------------------------
            # CALCULATE SPEED
            # ------------------------------------------------

            if track_id in vehicle_positions:


                previous_position = (
                    vehicle_positions[track_id]
                )


                previous_time = (
                    vehicle_times[track_id]
                )


                # Time difference
                time_difference = (
                    current_time -
                    previous_time
                )


                if time_difference > 0:


                    # ----------------------------------------
                    # PIXEL MOVEMENT
                    # ----------------------------------------

                    dx = (
                        current_position[0]
                        -
                        previous_position[0]
                    )


                    dy = (
                        current_position[1]
                        -
                        previous_position[1]
                    )


                    pixel_distance = math.sqrt(

                        dx ** 2 +
                        dy ** 2

                    )


                    # ----------------------------------------
                    # PIXELS PER SECOND
                    # ----------------------------------------

                    pixel_speed = (

                        pixel_distance /
                        time_difference

                    )


                    # ----------------------------------------
                    # PIXELS → METERS
                    # ----------------------------------------

                    meter_speed = (

                        pixel_speed *
                        PIXEL_TO_METER

                    )


                    # ----------------------------------------
                    # METERS/SECOND → KM/H
                    # ----------------------------------------

                    raw_speed_kmph = (

                        meter_speed *
                        3.6

                    )


                    # ----------------------------------------
                    # SPEED SMOOTHING
                    # ----------------------------------------

                    speed_history[
                        track_id
                    ].append(
                        raw_speed_kmph
                    )


                    speed_kmph = (

                        sum(
                            speed_history[
                                track_id
                            ]
                        )
                        /
                        len(
                            speed_history[
                                track_id
                            ]
                        )

                    )


            # ------------------------------------------------
            # SAVE CURRENT DATA
            # ------------------------------------------------

            vehicle_positions[
                track_id
            ] = current_position


            vehicle_times[
                track_id
            ] = current_time


            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2

            )


            # ------------------------------------------------
            # DRAW TRACKING POINT
            # ------------------------------------------------

            cv2.circle(

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


            cv2.putText(

                frame,

                text,

                (x1, y1 - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 255, 255),

                2

            )


    # ========================================================
    # DISPLAY VIDEO
    # ========================================================

    cv2.imshow(

        "Hikvision Vehicle Speed",

        frame

    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    key = cv2.waitKey(1)


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.stop()

cv2.destroyAllWindows()

print("Program stopped.")
