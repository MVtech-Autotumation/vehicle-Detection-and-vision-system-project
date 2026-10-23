from ultralytics import YOLO
import cv2 as c1
import numpy as np


model = YOLO("yolo11s.pt")
rtsp_url = r"D:\MVtech basic train model\traffic model1.1\traffic_SampleVideo_LowQuality.mp4"

webcam = c1.VideoCapture(rtsp_url) #testvideo.mp4

class_list = model.names

# DISTANCE CALIBRATION

# IMPORTANT:
# These are example calibration values.
# We will replace them with your REAL measurements.

# Image points (pixels)
# Select 4 points on the ROAD.
#
# Order:
# 1 = near-left
# 2 = near-right
# 3 = far-right
# 4 = far-left

source_points = np.float32([
    [150, 650],   # near-left
    [600, 650],   # near-right
    [470, 400],   # far-right
    [300, 400]    # far-left
])

# Real-world coordinates in meters
# Example road area:
#
# near-left  = (0, 0)
# near-right = (5, 0)
# far-right  = (5, 20)
# far-left   = (0, 20)

destination_points = np.float32([
    [0, 0],
    [5, 0],
    [5, 20],
    [0, 20]
])

# Calculate perspective transformation
matrix = c1.getPerspectiveTransform(
    source_points,
    destination_points
)


# MAIN LOOP


while True:

    ret, frame = webcam.read()

    if ret == True:

        #frame = c1.flip(frame, 1)

        # YOLO tracking
        results = model.track(
            frame,
            persist=True
        )

        if results[0].boxes.data is not None:

            boxes = results[0].boxes.xyxy.cpu()

            # Check if tracking IDs exist
            if results[0].boxes.id is not None:

                trackid = results[0].boxes.id.int().cpu().tolist()

                clsid = results[0].boxes.cls.int().cpu().tolist()

                confi = results[0].boxes.conf.cpu()

                
                # PROCESS EACH VEHICLE
                

                for box, track_id, class_idx, conf in zip(
                    boxes,
                    trackid,
                    clsid,
                    confi
                ):

                    x1, y1, x2, y2 = map(int, box)

                    class_name = class_list[class_idx]

                    
                    # FIND BOTTOM-CENTER OF VEHICLE
                    

                    center_x = int((x1 + x2) / 2)

                    bottom_y = y2

                    # Point touching the road
                    vehicle_point = np.float32([
                        [[center_x, bottom_y]]
                    ])

                
                    # CONVERT IMAGE POINT TO ROAD COORDINATE
                    

                    real_point = c1.perspectiveTransform(
                        vehicle_point,
                        matrix
                    )

                    real_x = real_point[0][0][0]

                    real_y = real_point[0][0][1]

                    # Distance from camera/origin
                    distance = np.sqrt(
                        real_x ** 2 +
                        real_y ** 2
                    )

                    
                    # DRAW VEHICLE
                    

                    c1.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Bottom-center point
                    c1.circle(
                        frame,
                        (center_x, bottom_y),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    
                    # DISPLAY INFORMATION
                    

                    text1 = f"{class_name}"

                    text2 = f"ID: {track_id}"

                    text3 = f"Distance: {distance:.1f} m"

                    c1.putText(
                        frame,
                        text1,
                        (x1, y1 - 35),
                        c1.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    c1.putText(
                        frame,
                        text2,
                        (x1, y1 - 15),
                        c1.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    c1.putText(
                        frame,
                        text3,
                        (x1, y2 + 25),
                        c1.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

       

        c1.imshow(
            "Vehicle Distance Detection",
            frame
        )

        key = c1.waitKey(1)

        if key == ord("q"):
            break

webcam.release()

c1.destroyAllWindows()