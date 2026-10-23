import cv2
import numpy as np


# RTSP CAMERA

video_path = "rtsp://admin:shree%40123@192.168.0.21:554/Streaming/channels/102"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot connect to Hikvision camera")
    exit()

print("Camera connected successfully")



# CALIBRATION DATA


# Real distance in centimeters
CALIBRATION_DISTANCE = np.array([
    50,
    70,
    80,
    100,
    127
], dtype=np.float32)

# Red square pixel width measured at those distances
CALIBRATION_PIXELS = np.array([
    31,
    24,
    21,
    20,
    13
], dtype=np.float32)


# DISTANCE CALCULATION


def calculate_distance(pixel_width):

    if pixel_width <= 0:
        return None

    # Pixel size decreases as distance increases.
    # Reverse arrays so np.interp receives increasing x values.

    pixels = CALIBRATION_PIXELS[::-1]
    distances = CALIBRATION_DISTANCE[::-1]

   
    # Don't trust values outside calibration range
    

    if pixel_width < pixels[0] or pixel_width > pixels[-1]:
        return None

    distance = np.interp(
        pixel_width,
        pixels,
        distances
    )

    return distance



# RED COLOR DETECTION


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read frame")
        break

    
    # Convert BGR → HSV
   

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    # RED COLOR RANGE
    

    lower_red1 = np.array([0, 70, 30])
    upper_red1 = np.array([12, 255, 220])

    lower_red2 = np.array([170, 70, 30])
    upper_red2 = np.array([180, 255, 220])

    mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    mask = mask1 | mask2

    
    # CLEAN MASK
    

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    
    # FIND RED OBJECTS
    

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_area = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny noise
        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        
        # SQUARE CHECK
        

        aspect_ratio = w / float(h)

        if not (0.75 <= aspect_ratio <= 1.25):
            continue

       
        # Select largest valid red square
        

        if area > best_area:

            best_area = area
            best_contour = contour

    
    # PROCESS DETECTED RED SQUARE
    

    if best_contour is not None:

        x, y, w, h = cv2.boundingRect(best_contour)

        # Center
        center_x = x + w // 2
        center_y = y + h // 2

        
        # Calculate distance
       

        pixel_width = float(w)

        distance = calculate_distance(pixel_width)

        # Draw bounding box

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Center point
        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        # RED SQUARE LABEL

        cv2.putText(
            frame,
            "RED SQUARE",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        
        # PIXEL SIZE

        cv2.putText(
            frame,
            f"Size: {w} x {h} px",
            (x, y + h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        
        # CENTER POINT

        cv2.putText(
            frame,
            f"Point: ({center_x}, {center_y})",
            (x, y + h + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

       
        # DISTANCE
        

        if distance is not None:

            cv2.putText(
                frame,
                f"Distance: {distance:.1f} cm",
                (x, y + h + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        else:

            # Outside calibration range
            cv2.putText(
                frame,
                "Distance: OUT OF RANGE",
                (x, y + h + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

   

    cv2.imshow(
        "Hikvision Red Square Distance",
        frame
    )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()