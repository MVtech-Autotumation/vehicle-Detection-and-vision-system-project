import cv2 as c1  # Import library
webcam=c1.VideoCapture(0)
while True:
    ret,frame=webcam.read()

    if ret==True:
        c1.imshow("video playing ", frame)
        key=c1.waitKey(1)
        if key == ord("q"):
            break


webcam.release()


c1.destroyAllWindows()
# Close all OpenCV windows