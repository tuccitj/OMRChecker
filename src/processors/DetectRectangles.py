from itertools import groupby
from pprint import pprint
import cv2
from matplotlib import pyplot as plt
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor

class DetectRectangles(ImagePreprocessor):
    """Detect rectangles and draws their boundaries"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_filter(self, image, filename):
        """Apply filter to the image and returns modified image"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # Convert the image to grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # Determine optimal threshold using Otsu's method
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Apply edge detection to the grayscale image
        edged = cv2.Canny(thresh, 10, 200)
        # Find contours in the edge map
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Initialize a list to store the coordinates of the rectangles
        rect_coords = []
        # Iterate through the contours and filter for rectangles
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w > 50 and h > 50 and w < 1000 and h < 1000:  # Only consider rectangles with a width and height greater than 50 pixels
                    rect_coords.append((x, y, w, h))
        rect_coords.sort(key=lambda x: x[0])        
        tolerance = 10
        # group coordinates by x value with tolerance and sort each group by y value
        grouped_coordinates = []
        for key, group in groupby(rect_coords, lambda x: round(x[0]/tolerance)):
            grouped_coordinates.extend(sorted(list(group), key=lambda x: x[1]))
        # Draw the rectangles on the image in the sorted order
        for i, (x, y, w, h) in enumerate(grouped_coordinates):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = "{}".format(i+1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (x, y - 5), font, 0.5, (0, 0, 255), 2)
        image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
        return(image)

    @staticmethod
    def exclude_files():
        """Returns a list of file paths that should be excluded from processing"""
        return []
