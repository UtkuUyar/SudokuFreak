import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


class GridDetector:
    GRID_WIDTH = 400  # Width value for final result
    GRID_HEIGHT = 400  # Height value for final result

    def __init__(self, imagePath):
        self.path = imagePath

    def loadImage(self, filename):
        self.image = cv2.imread(os.path.join(self.path, filename))
        self.image = cv2.resize(
            self.image, (self.GRID_WIDTH, self.GRID_HEIGHT))

    def preProcessing(self):
        processed = cv2.cvtColor(
            self.image, cv2.COLOR_BGR2GRAY)  # Apply grayscale
        # Add some blur for more smooth input for binary thresholding
        processed = cv2.GaussianBlur(processed, (5, 5), 1)
        # Apply binary inverse thresholding
        processed = cv2.adaptiveThreshold(processed, 255, 1, 1, 11, 2)

        self.preprocessed = processed
        return self.preprocessed

    def findBiggestContour(self, contours, epsilon_multiplier=0.06):
        biggest = np.array([])
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 75:
                circumference = cv2.arcLength(contour, True)
                # Detect num. of corners for this contour. If not rectangular, skip.
                approxCorner = cv2.approxPolyDP(
                    contour, epsilon_multiplier * circumference, True)
                if area > max_area and len(approxCorner) == 4:
                    biggest = approxCorner
                    max_area = area
        return {"contour": self.reorder(biggest), "area": max_area}

    def reorder(self, contour):
        # Function for reordering corners of a contour as [top-left, top-right, bottom-left, bottom-right]
        points = contour.reshape(4, -1)
        reordered = np.zeros((4, 1, 2), dtype=int)
        # Find the top-left and bottom-right corners by the sum of their coordinate values
        sumOrder = np.argsort(np.sum(points, axis=1))
        reordered[0] = points[sumOrder[0]]
        reordered[3] = points[sumOrder[3]]

        # Find the top-right and bottom-left corners by their y coordinates
        verticalOrder = np.argsort(np.diff(points, axis=1).reshape(1, -1))
        reordered[1] = contour[verticalOrder[0][0]]
        reordered[2] = contour[verticalOrder[0][3]]

        return reordered

    def findContours(self):
        return cv2.findContours(
            self.preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def birdEyeView(self, region, maxArea):
        if region.size != 0 and maxArea > 0:
            # Setting up mapping for the transformation
            pts1 = region.astype(np.float32)
            pts2 = np.array([[0, 0], [self.GRID_WIDTH, 0], [0, self.GRID_HEIGHT], [
                            self.GRID_WIDTH, self.GRID_HEIGHT]], dtype=np.float32)
            # Calculate the transformation
            transform = cv2.getPerspectiveTransform(pts1, pts2)
            # Apply transformation to original image.
            transformed_image = cv2.warpPerspective(
                self.image, transform, (self.GRID_WIDTH, self.GRID_HEIGHT))
            return transformed_image

    def run(self):
        self.preProcessing()
        contours, hierarchy = self.findContours()
        allContours = self.image.copy()
        cv2.drawContours(allContours, contours, -1, (0, 0, 255), 4)

        cv2.imshow("image36_contours", allContours)
        cv2.waitKey()

        biggestContour = self.findBiggestContour(contours)
        grid = self.birdEyeView(
            biggestContour["contour"], biggestContour["area"])
        return grid, biggestContour["contour"]


if __name__ == "__main__":
    MAIN_PATH = "datasets_rearranged/train/images/"

    Detector = GridDetector(MAIN_PATH)
    # random_sudokus = np.random.randint(0, 160, size=(10,))
    # random_sudokus = np.arange(0, 160)
    # filenames = os.listdir(MAIN_PATH)

    # for i in random_sudokus:
    #     Detector.loadImage(filenames[i])
    #     image, original_corners = Detector.run()
    #     cv2.imshow(f"{filenames[i]}", image)
    #     cv2.waitKey()

    Detector.loadImage("image36.jpg")
    image, _ = Detector.run()
    cv2.imshow("image36", image)
    cv2.waitKey()
