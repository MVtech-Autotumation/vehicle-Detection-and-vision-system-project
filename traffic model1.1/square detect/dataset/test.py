import cv2
import numpy as np

# =====================================================
# VIDEO
# =====================================================

video_path = r"C:\Users\User\Videos\Screen Recordings\Screen Recording 2026-10-02 170242.mp4"

camera = cv2.VideoCapture(video_path)

if not camera.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")


while True:

    ret, frame = camera.read()

    if not ret:
        break

    # =================================================
    # HSV
    # =================================================

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # =================================================
    # RED COLOR
    # =================================================

    lower_red1 = np.array([0, 120, 40])
    upper_red1 = np.array([15, 255, 255])

    lower_red2 = np.array([165, 120, 40])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

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
    # FIND CONTOURS
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

        # Red square should be approximately square
        ratio = w / float(h)

        if ratio < 0.6 or ratio > 1.5:
            continue

        if area > best_area:
            best_area = area
            best = (x, y, w, h)

    # =================================================
    # DETECTED RED SQUARE
    # =================================================

    if best is not None:

        x, y, w, h = best

        # Center
        center_x = x + w // 2
        center_y = y + h // 2

        # Bounding box
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
            6,
            (0, 0, 255),
            -1
        )

        # Label
        cv2.putText(
            frame,
            "red_square",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Coordinates
        cv2.putText(
            frame,
            f"Point: ({center_x}, {center_y})",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # =================================================
    # SHOW ONLY ORIGINAL VIDEO
    # =================================================

    cv2.imshow(
        "Red Square Detection",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()