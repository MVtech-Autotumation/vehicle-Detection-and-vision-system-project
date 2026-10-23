import cv2

img = cv2.imread(r"D:\MVtech basic train model\test\rust.jpg")

#--------------------------------------
#  IMPORTANT NOTE ONLY PASS JPG IMAGE
#----------------------------------------
img2 = cv2.resize(img,(500,500))

cv2.imshow("rust", img2)

cv2.waitKey(0)

cv2.destroyAllWindows()