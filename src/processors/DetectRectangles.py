import cv2
from matplotlib import pyplot as plt
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor

class DetectRectangles(ImagePreprocessor):
    """Detect rectangles and draws their boundaries"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_filter(self, image, filename):
        """Apply filter to the image and returns modified image"""
        #TODO add handling for filtering by rectangle min and max size, 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # Convert the image to grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # Determine optimal threshold using Otsu's method
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Apply edge detection to the grayscale image
        edged = cv2.Canny(thresh, 10, 200)
        # plt.title("Edged")
        # plt.imshow(edged)
        # plt.show()
        # Find contours in the edge map
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a list to store the coordinates of the rectangles
        rect_coords = []

        # Iterate through the contours and filter for rectangles
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                rect_coords.append((x, y, w, h))

        # Sort the rectangles by their y-coordinate first and then by their x-coordinate
        rect_coords = sorted(rect_coords, key=lambda x: (x[1], x[0]))

        # Draw the rectangles on the image in the sorted order
        for i, (x, y, w, h) in enumerate(rect_coords):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print("Rectangle %d: (x=%d, y=%d, w=%d, h=%d)" % (i+1, x, y, w, h))

        # Show the detected rectangles on the original image
        # plt.title("Final")
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.show()
        ## change back to grayscale for next process
        image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
        return(image)

    @staticmethod
    def exclude_files():
        """Returns a list of file paths that should be excluded from processing"""
        return []
