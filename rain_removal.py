import cv2
import numpy as np

image_path = "raindrop_image.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

kernel = np.ones((8, 8), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=0)

result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

cv2.imwrite("raindrop_removed.jpeg", result)
cv2.imshow("Original", image)
cv2.imshow("Rain Removed", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Rain removed and image saved as 'raindrop_removed.jpeg'")
