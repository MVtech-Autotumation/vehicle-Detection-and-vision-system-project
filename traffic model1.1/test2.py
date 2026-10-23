import cv2 as c1

webcam = c1.VideoCapture("traffic_SampleVideo_LowQuality.mp4")

while True:
    ret, frame = webcam.read()

    if not ret:
        break

    c1.imshow("Original Video", frame)

    # Press q to stop
    if c1.waitKey(25) & 0xFF == ord('q'):
        break

webcam.release()
c1.destroyAllWindows()