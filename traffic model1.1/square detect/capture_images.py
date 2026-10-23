import cv2
import os

# =====================================================
# HIKVISION CAMERA
# =====================================================

rtsp_url = "rtsp://admin:shree@123@192.168.0.21:554/Streaming/channels/101"

camera = cv2.VideoCapture(rtsp_url)

if not camera.isOpened():
    print("Camera connection failed")
    exit()

print("Camera connected")

# =====================================================
# CREATE FOLDER
# =====================================================

os.makedirs("dataset/images/all", exist_ok=True)

count = 0

print()
print("Controls:")
print("SPACE = Capture image")
print("Q     = Quit")
print()

while True:

    ret, frame = camera.read()

    if not ret:
        print("Frame error")
        break

    cv2.putText(
        frame,
        f"Images captured: {count}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Capture Red Square Images", frame)

    key = cv2.waitKey(1) & 0xFF

    # -------------------------------------------------
    # CAPTURE
    # -------------------------------------------------

    if key == 32:  # SPACE

        filename = f"dataset/images/all/image_{count:04d}.jpg"

        cv2.imwrite(filename, frame)

        print("Saved:", filename)

        count += 1

    # -------------------------------------------------
    # EXIT
    # -------------------------------------------------

    elif key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print()
print(f"Total images captured: {count}")