import cv2
import numpy as np

img_color = cv2.imread("image.jpeg")


print("Color Image:", img_color.shape)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
print("Gray Image:", img_gray.shape)

img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

_, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
_, binary_inv = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

adaptive = cv2.adaptiveThreshold(
    img_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)


median = cv2.medianBlur(img_gray, 5)

bilateral = cv2.bilateralFilter(img_color, 9, 75, 75)

sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)

img_canny = cv2.Canny(img_blur, 100, 200)

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

line_image = img_color.copy()
cv2.line(line_image, (0, 0), (300, 300), (255, 0, 0), 5)

cv2.imshow("Original", img_color)
cv2.imshow("Gray", img_gray)
cv2.imshow("Gaussian Blur", img_blur)
cv2.imshow("Binary", binary)
cv2.imshow("Binary Inverse", binary_inv)
cv2.imshow("Adaptive Threshold", adaptive)
cv2.imshow("Median Blur", median)
cv2.imshow("Bilateral Filter", bilateral)
cv2.imshow("Sobel X", sobelx)
cv2.imshow("Sobel Y", sobely)
cv2.imshow("Canny", img_canny)
cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)
cv2.imshow("Line Image", line_image)

cv2.waitKey(0)
cv2.destroyAllWindows()