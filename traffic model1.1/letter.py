import cv2
import numpy as np

# =====================================================
# VIDEO
# =====================================================

video_path = "rtsp://admin:shree%40123@192.168.0.21:554/Streaming/channels/102"

camera = cv2.VideoCapture(video_path)

if not camera.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")


# =====================================================
# KNOWN OBJECT INFORMATION
# =====================================================

REAL_SQUARE_SIZE_CM = 3.5

# The square is currently known to be 120 cm away
CALIBRATION_DISTANCE_CM = 120.0


# =====================================================
# CALIBRATION
# =====================================================

focal_length = None


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = camera.read()

    if not ret:
        break

    # =================================================
    # RED DETECTION
    # =================================================

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 40])
    upper_red1 = np.array([15, 255, 255])

    lower_red2 = np.array([165, 120, 40])
    upper_red2 = np.array([180, 255, 255])

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


    # =================================================
    # CLEAN MASK
    # =================================================

    kernel = np.ones((5, 5), np.uint8)

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


    # =================================================
    # FIND RED SQUARE
    # =================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 200:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 15 or h < 15:
            continue

        ratio = w / float(h)

        if ratio < 0.6 or ratio > 1.5:
            continue

        if area > best_area:

            best_area = area
            best = (x, y, w, h)


    # =================================================
    # RED SQUARE FOUND
    # =================================================

    if best is not None:

        x, y, w, h = best

        # Center
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Draw center
        cv2.circle(
            frame,
            (center_x, center_y),
            6,
            (0, 0, 255),
            -1
        )

        # Label
        cv2.putText(
            frame,
            "RED SQUARE",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # =================================================
        # PIXEL SIZE
        # =================================================

        pixel_size = (w + h) / 2


        # =================================================
        # AUTOMATIC CALIBRATION
        # =================================================

        if focal_length is None:

            focal_length = (
                pixel_size * CALIBRATION_DISTANCE_CM
            ) / REAL_SQUARE_SIZE_CM

            print("--------------------------------")
            print("Automatic calibration completed")
            print(f"Pixel size : {pixel_size:.2f} px")
            print(
                f"Known distance : "
                f"{CALIBRATION_DISTANCE_CM:.2f} cm"
            )
            print(
                f"Focal value : "
                f"{focal_length:.2f}"
            )
            print("--------------------------------")


        # =================================================
        # DISTANCE CALCULATION
        # =================================================

        distance_cm = (
            focal_length * REAL_SQUARE_SIZE_CM
        ) / pixel_size


        # =================================================
        # DISPLAY DISTANCE
        # =================================================

        cv2.putText(
            frame,
            f"Distance: {distance_cm:.1f} cm",
            (x, y + h + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # =================================================
        # DISPLAY PIXEL SIZE
        # =================================================

        cv2.putText(
            frame,
            f"Size: {w} x {h} px",
            (x, y + h + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )


    else:

        cv2.putText(
            frame,
            "RED SQUARE NOT DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # =================================================
    # DISPLAY
    # =================================================

    cv2.imshow(
        "Red Square Distance",
        frame
    )


    # =================================================
    # QUIT
    # =================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =====================================================
# RELEASE
# =====================================================

camera.release()
cv2.destroyAllWindows()