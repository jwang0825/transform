import cv2
import numpy as np

def detect_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 25)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=40, param2=30, minRadius=75, maxRadius=150)

    if circles is not None:

        circles = np.uint16(np.around(circles))

        for (x, y, r) in circles[0, :1]:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)

def get_perspective():
	pts1 = np.float32([[770, 320], [1660, 100], [900, 800], [1770, 600]])

	pts2 = np.float32([[1090, 295], [1380, 295], [1090, 590], [1380, 590]])

	return cv2.getPerspectiveTransform(pts1, pts2)

if __name__ == "__main__":
	image = cv2.imread('test-l3.jpeg', cv2.IMREAD_COLOR)

	height = image.shape[0]
	width = image.shape[1]

	cv2.imshow("original", image) 

	matrix = get_perspective()

	# Get the perspective of the original image
	trans = cv2.warpPerspective(image, matrix, (width, height))

	# detect circle in the transformed image
	detect_circle(trans)

	# Restore to original image with circle detected
	# Take the inverse of the matrix used to transform the image
	inverse_matrix = np.linalg.inv(matrix)
	result = cv2.warpPerspective(trans, inverse_matrix, (width, height))
	cv2.imshow("result", result) 
	cv2.imwrite("updated_image.jpeg",result)
	cv2.waitKey(0)
