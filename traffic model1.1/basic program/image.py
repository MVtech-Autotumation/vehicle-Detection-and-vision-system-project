import cv2 as c1  # Import library

img = c1.imread(r"C:\Users\User\Downloads\images.jpg")
# imread() is used to read the image

img1 = c1.resize(img, (1200, 700))
# resize() is used to resize the image

c1.imshow("original image", img1)
# imshow() is used to display the image

c1.waitKey(0)
# Wait until a key is pressed

c1.destroyAllWindows()
# Close all OpenCV windows