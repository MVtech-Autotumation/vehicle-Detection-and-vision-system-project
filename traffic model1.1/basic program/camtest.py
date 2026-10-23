import cv2

# Hikvision RTSP stream
rtsp_url = "rtsp://admin:shree@123@192.168.0.21:554/Streaming/channels/101"

camera = cv2.VideoCapture(rtsp_url)

if not camera.isOpened():
    print("ERROR: Could not connect to Hikvision camera")
    exit()

print("Camera connected successfully!")

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read frame")
        break

    cv2.imshow("Hikvision Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()