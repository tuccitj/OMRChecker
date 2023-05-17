import itertools
import multiprocessing
from pprint import pprint
import time
from itertools import groupby
import timeit
import cv2
import numpy as np
from matplotlib import pyplot as plt
import src.constants as constants
import src.custom.gridfinder as gridfinder
from src.logger import logger
from sklearn.cluster import KMeans

def rgb(red, green, blue):
    return red, green, blue
class DrawConfigs:
    """Repository of configurations for use with cv2 functions such as DrawContours(), PolyLines(), and PutText()"""

    class DrawConfig:

        def __init__(self, fontFace, fontScale, color, thickness):
            self.fontFace = fontFace
            self.fontScale = fontScale
            self.color = color
            self.thickness = thickness

    DEFAULT = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale=1,
                         color=rgb(0, 0, 255),
                         thickness=2)
    DEFAULT_LINE = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=1,
                              color=rgb(0, 0, 255),
                              thickness=2)
    DEFAULT_LABEL = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=1,
                               color=rgb(0, 0, 255),
                               thickness=2)
    UPPER_LEFT_LARGE_LABEL = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=5,
                                        color=rgb(0, 0, 255),
                                        thickness=5)
    IMG_PROC_TEMPLATE = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                   fontScale=1,
                                   color=rgb(0, 0, 0),
                                   thickness=-1)
    LARGE_FONT_IN_WHITE = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=5,
                                     color=rgb(0, 0, 255),
                                     thickness=5)
    DRAW_ON_BLANK = DrawConfig(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=rgb(255, 255, 255),
        thickness=-1,
    )

    DRAW_BLACK_TEXT = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=5,
                                 color=(rgb(0, 0, 0)),
                                 thickness=-1)
    DRAW_ON_TRANSPARENT = DrawConfig(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=rgb(238, 255, 0),
        thickness=3,
    )
    DRAW_TRANSPARENT_BOX = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=1,
                                      color=rgb(0, 36, 14),
                                      thickness=-1)

class DetectionMethod:
    METHOD_1 = 1
    METHOD_2 = 2
    METHOD_3 = 3
    METHOD_4 = 4
    METHOD_5 = 5
    METHOD_6 = 6
    METHOD_7 = 7
class ShapeSortMethods:
    DEFAULT = 0
    LEFT_TO_RIGHT_TOP_TO_BOTTOM = 1
    TOP_TO_BOTTOM_LEFT_TO_RIGHT = 2
    TEST_2 = 3,
    TEST_L2R = 4
class CodeTimer:
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''
    def __enter__(self):
        # end_time = timeit.default_timer()
        self.start_time = timeit.default_timer()
    def __exit__(self, exc_type, exc_value, traceback):
        # self.end_time = (timeit.default_timer() - self.start) * 1000.0
        self.end_time = timeit.default_timer()
        execution_time = self.end_time - self.start_time
        logger.info(f"Execution time for {self.name}: {execution_time:.6f} seconds")

class Shape():
    """Smallest unit for processing. A shape is usually within a ShapeArray
    which represents a Field Block in the template.
    """

    def __init__(
        self,
        contour,
        draw_line_config=DrawConfigs.DEFAULT_LINE,
        draw_label_config=DrawConfigs.DEFAULT_LABEL,
    ):
        self.contour = contour
        self.image = None
        self.vertices = self._get_vertices(contour, sort=True)
        self.draw_line_config = draw_line_config
        self.draw_label_config = draw_label_config
        self.detection_boxes = None  # This is currently a shape array... potentially inherit functionality from
        self.processing_mask = None  # used
        self.isMarked = False
        self.value = None  # Only in DetectionBox context

    def draw(self,
             image,
             label_shape=False,
             label="",
             draw_line_config=None,
             draw_label_config=None,
             display_image=False,
             transparent_box=False):
        # Set DrawConfigs to defaults
        if not draw_line_config:
            draw_line_config = self.draw_line_config
        if not draw_label_config:
            draw_label_config = self.draw_label_config

        if transparent_box == True:
            transp_layer = image.copy()
            final_marked = image.copy()
            # Overlay Transparencies
            alpha = 0.65
            # Draw on transparency layer
            cv2.rectangle(
                final_marked,
                self.vertices[0],
                self.vertices[2],
                constants.CLR_DARK_GRAY,
                -1,
            )

            # Translucent
            cv2.addWeighted(final_marked, alpha, transp_layer, 1 - alpha, 0,
                            image)
            # Draw on image after transparency layer
            cv2.putText(
                image,
                label,
                self.label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                constants.TEXT_SIZE + 1,
                rgb(255, 255, 255),
                int(1 + 3.5 * constants.TEXT_SIZE),
            )

        else:
            # Draw the contours in the image
            cv2.drawContours(
                image,
                [self.contour],
                -1,
                draw_line_config.color,
                draw_line_config.thickness,
            )
            if label_shape:
                self.label(image, label, draw_label_config)

        # identify the label position
        # self.label_position = (self.vertices[0][0], self.vertices[0][1])
        if display_image:
            plshow("shape", image)
        return image

    def label(self, image, label, draw_config):
        if draw_config:
            cv2.putText(
                image,
                label,
                self.label_position,
                draw_config.fontFace,
                draw_config.fontScale,
                draw_config.color,
                draw_config.thickness,
            )
        else:
            raise ValueError("No draw configuration provided")
        return image

    def process_detection_boxes(self):
        for idx, detection_box in enumerate(self.detection_boxes):
            detection_box.isMarked = detection_box._meetsBlackThreshold(
                sub_image=self.processing_mask, debug_level=0)

    def _get_vertices(self, contour, sort):
        rect = cv2.minAreaRect(contour)
        vertices = np.int0(cv2.boxPoints(rect))
        self.centroid = np.round(np.mean(vertices, axis=0)).astype(int)
        self.label_position = (self.centroid[0] - 10, self.centroid[1] + 15)
        #! Deprecated in favor of np.mean
        # Calculate centroid coordinates
        # M = cv2.moments(cnt)
        # cx = int(M['m10'] / M['m00'])
        # cy = int(M['m01'] / M['m00'])
        # # Set label position to centroid coordinates
        # label_position = (cx - 10, cy + 15)
        #! Deprecated in favor of minAreaRect...needed to guarantee 4 vertices.
        # approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        # vertices = np.int0(
        #     [approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
        # print(f"Original: {vertices}")
        if sort:
            vertices = self._sort_vertices(vertices)
        return vertices

    def _sort_vertices(self, box):
        # Get the indices that would sort the array by y value
        sorted_y_indices = np.argsort(box[:, 1])
        # Split the array into two lists based on the y values
        lowest_y_list = box[sorted_y_indices[:2]].tolist()
        highest_y_list = box[sorted_y_indices[2:]].tolist()
        # Sort lowest_y_list by the lowest x value to the highest x value
        sorted_lowest_y_list = sorted(lowest_y_list,
                                      key=lambda point: point[0])
        # Sort highest_y_list by the highest x value to the lowest x value
        sorted_highest_y_list = sorted(highest_y_list,
                                       key=lambda point: point[0],
                                       reverse=True)
        sorted_lowest_y_list.extend(sorted_highest_y_list)
        box = sorted_lowest_y_list.copy()
        box = np.int0(box)
        return box

    def _meetsBlackThreshold(self, sub_image, threshold=0.5, debug_level=0):
        """Processes a sub-image at a shape level, assuming that the input is a grayscale
        image where the detection box areas should have non-zero values. The zero
        values are used to determine if the black threshold is met.

        Args:
            sub_image (numpty.ndarray): The input sub-image.

            threshold (float, optional): The threshold value for black pixels ratio. Defaults to 0.5.

            debug_level (int, optional): The level of debugging information to display.
                Set to 0 for no debugging information, 1 for basic debugging information,
                and 2 for full debugging information. Defaults to 0.

        Returns:
            bool: True if the ratio of black pixels within the detection box is greater
                than the given threshold, False otherwise.
        """
        if debug_level == 1:
            plshow("Inputted Image", sub_image)
            mask = np.zeros_like(sub_image)
            plshow("mask", mask)
            cv2.drawContours(mask, [self.contour], 0, 255, -1)
            plshow("mask w/drawn detection box", mask)
            masked_image = cv2.bitwise_and(sub_image, mask)
            plshow("masked image", masked_image)
        else:
            mask = np.zeros_like(sub_image)
            cv2.drawContours(mask, [self.contour], 0, 255, -1)
            masked_image = cv2.bitwise_and(sub_image, mask)
        # Count the number of black pixels within the detection box area
        num_black_pixels = cv2.countNonZero(masked_image)
        # Calculate the area of the detection box
        box_area = cv2.contourArea(self.contour)
        # Calculate the ratio of black pixels to the total area
        black_ratio = num_black_pixels / box_area
        # Check if the black ratio is greater than the threshold
        return black_ratio > threshold

class ShapeArray():
    """Array of Shapes"""

    def __init__(self, shapes):
        self.shapes = shapes
        self.labels = [["p", "b", "k", "q", "n", "r", "0-0", "-0", "X"],
                       ["a", "b", "c", "d", "e", "f", "g", "h", "+"],
                       ["1", "2", "3", "4", "5", "6", "7", "8", "="]]
        self.labels = list(itertools.chain(*self.labels))
        # if sort_method:
        #     self.shapes = self._sort_shapes(sort_method)

    def __getitem__(self, index):
        return self.shapes[index]

    def __len__(self):
        return len(self.shapes)

    def sort_shapes(self, sort_method, tolerance=30):
        return ShapeArray(self._sort_shapes(sort_method, tolerance))

    #? Should this be a public method so we can sort the ShapeArray on demand? Possibly helpful for multiprocessing
    #? Additionally, we need to assign an ID to each shape once they're in the proper order...I'd like to avoid another iteration
    def _sort_shapes(self, sort_method, tolerance=30):
        """Method for sorting shapes in a FieldBlock

        Args:
            sort_method (_type_): Sort method to sort by
            tolerance (int, optional): _description_. Defaults to 30.

        Returns:
            _type_: _description_
        """
        shapes = self.shapes
        shapes_final = []
        if sort_method == ShapeSortMethods.TOP_TO_BOTTOM_LEFT_TO_RIGHT:
            #sort by x-value
            sorted_shapes = sorted(shapes,
                                   key=lambda shape: shape.vertices[0][0])
            # TODO tolerance should either be determined mathmatically w/distributions or set as a constant in config
            tolerance = tolerance
            # group coordinates by x value with tolerance and sort each group by y value
            for key, group in groupby(
                    sorted_shapes,
                    lambda shape: round(shape.vertices[0][0] / tolerance)):
                shapes_final.extend(
                    sorted(list(group),
                           key=lambda shape: shape.vertices[0][1]))
        elif sort_method == ShapeSortMethods.LEFT_TO_RIGHT_TOP_TO_BOTTOM:
            detection_boxes = self.shapes
            # Get the list of y values in the upper left vertex for each detection box
            y_vals = np.array([box.vertices[0][1] for box in detection_boxes])
            # Use k-means clustering to group the detection boxes by y values
            n_clusters = 3
            reshaped_y_vals = y_vals.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters,
                            random_state=0,
                            n_init='auto').fit(reshaped_y_vals)
            # Get the indices of the detection boxes in each cluster
            cluster_indices = [
                np.where(kmeans.labels_ == i)[0] for i in range(n_clusters)
            ]
            # Get the detection boxes in each cluster
            clustered_boxes = [[detection_boxes[i] for i in indices]
                               for indices in cluster_indices]
            # Sort the detection boxes in each row by their x-values
            for row in clustered_boxes:
                row.sort(key=lambda shape: shape.vertices[0][0])
            # Sort the rows by the y-value of their first detection box
            clustered_boxes.sort(key=lambda row: row[0].vertices[0][1])
            # flatten
            flattened_boxes = [box for row in clustered_boxes for box in row]
            # concatenated_boxes = np.concatenate((row1, row2, row3), axis=0)
            return flattened_boxes
        elif sort_method == ShapeSortMethods.TEST_L2R:
            #https://stackoverflow.com/questions/29630052/ordering-coordinates-from-top-left-to-bottom-right
            #detect the keypoints
            params = cv2.SimpleBlobDetector_Params()
            # params.minThreshold = 1
            # params.maxThreshold = 256
            # params.filterByArea = True
            # params.minArea = 350
            # params.filterByConvexity = False
            # params.filterByInertia = False
            detector = cv2.SimpleBlobDetector_create(params)
            img = ""
            keypoints = detector.detect(img)
            img_with_keypoints = cv2.drawKeypoints(
                img, keypoints, np.array([]), (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            points = []
            keypoints_to_search = keypoints[:]
            while len(keypoints_to_search) > 0:
                a = sorted(keypoints_to_search,
                           key=lambda p: (p.pt[0]) +
                           (p.pt[1]))[0]  # find upper left point
                b = sorted(keypoints_to_search,
                           key=lambda p: (p.pt[0]) -
                           (p.pt[1]))[-1]  # find upper right point

                cv2.line(img_with_keypoints, (int(a.pt[0]), int(a.pt[1])),
                         (int(b.pt[0]), int(b.pt[1])), (255, 0, 0), 1)

                # convert opencv keypoint to numpy 3d point
                a = np.array([a.pt[0], a.pt[1], 0])
                b = np.array([b.pt[0], b.pt[1], 0])

                row_points = []
                remaining_points = []
                for k in keypoints_to_search:
                    p = np.array([k.pt[0], k.pt[1], 0])
                    d = k.size  # diameter of the keypoint (might be a theshold)
                    dist = np.linalg.norm(
                        np.cross(np.subtract(p, a), np.subtract(
                            b, a))) / np.linalg.norm(
                                b)  # distance between keypoint and line a->b
                    if d / 2 > dist:
                        row_points.append(k)
                    else:
                        remaining_points.append(k)

                points.extend(sorted(row_points, key=lambda h: h.pt[0]))
                keypoints_to_search = remaining_points
        elif sort_method == ShapeSortMethods.TEST_2:
            detection_boxes = self.shapes
            # Get the list of y values in the upper left vertex for each detection box
            y_vals = np.array([box.vertices[0][1] for box in detection_boxes])
            # Use k-means clustering to group the detection boxes by y values
            n_clusters = 3
            reshaped_y_vals = y_vals.reshape(-1, 1)

            kmeans = KMeans(n_clusters=n_clusters,
                            random_state=0,
                            n_init='auto').fit(reshaped_y_vals)
            detection_boxes = np.array(self.shapes)
            # Get the detection boxes in each cluster
            clustered_boxes = [
                detection_boxes[kmeans.labels_ == i] for i in range(n_clusters)
            ]
            # Sort the detection boxes in each row by their x-values
            for i in range(n_clusters):
                clustered_boxes[i] = clustered_boxes[i][np.argsort(
                    [box.vertices[0][0] for box in clustered_boxes[i]])]
            # Sort the rows by the y-value of their first detection box
            clustered_boxes.sort(key=lambda row: row[0].vertices[0][1])
            # Flatten the list
            flattened_boxes = [box for row in clustered_boxes for box in row]

            # concatenated_boxes = np.concatenate((row1, row2, row3), axis=0)
            return flattened_boxes

            # TODO tolerance should be a parameter...should be able to set defaults somewhere
            sorted_shapes = sorted(shapes,
                                   key=lambda shape: shape.vertices[0][1])
            # TODO tolerance should either be determined mathmatically w/distributions or set as a constant in config
            # group coordinates by x value with tolerance and sort each group by y value
            for key, group in groupby(
                    sorted_shapes,
                    lambda shape: round(shape.vertices[0][1] / tolerance)):
                shapes_final.extend(
                    sorted(list(group),
                           key=lambda shape: shape.vertices[0][0]))
        # shapes_final=ShapeArray(shapes_final)
        return shapes_final

    def draw(
        self,
        image,
        label_shapes=False,
        draw_line_config=None,
        draw_label_config=None,
        display_image=False,
    ):
        for idx, shape in enumerate(self.shapes):
            label = "{}".format(idx + 1)
            if draw_line_config:
                shape.draw_line_config = draw_line_config
            if draw_label_config:
                shape.draw_label_config = draw_label_config
            # this is drawing each contour separately

            image = shape.draw(image, label_shape=label_shapes, label=label)
            # if label_shapes:
            #     label = "{}".format(idx+1)
            #     shape.draw(image, label_shapes=True, label = label)
            #     shape.label(image, label, font)
            if display_image:
                print("Hello World! Look at the different co")
                print("Welcome to emerald world!")
                plshow("ShapeArray", image)
        return image

    def multi_process_all_shapes(self, src_image, proc_template_method,
                                 box_detection_method, debug_level):
        """Process all shapes in parallel using multiprocessing.

        Args:
            src_image (numpy.ndarray): The source image to process.
            proc_template_method (type): The method for processing the template.
            box_detection_method (type): The method for detecting boxes.
            debug_level (type): The level of debugging.

        Returns:
            numpy.ndarray: The processed image.
        """
        template_config = None
        blank_image = np.zeros_like(src_image)
        blank_image.fill(255)
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
        # Generate a list of (index, shape) tuples to be processed by `process_shape()`.
        idx_to_debug = []  # indexes of shapes to enable debugging for
        shape_args = []  # list to store arguments for each shape
        for idx, shape in enumerate(self.shapes):
            if idx in idx_to_debug:
                # enable debugging for this shape
                debug_level = 1
            else:
                # disable debugging for this shape
                debug_level = debug_level
            shape_arg = (idx, shape, src_image, proc_template_method,
                         box_detection_method, debug_level)
            shape_args.append(shape_arg)
        run = 0
        if run == 1:
            shape_args = [(idx, shape, src_image, proc_template_method,
                           box_detection_method, debug_level)
                          for idx, shape in enumerate(self.shapes)]
        # Use `map_async()` to execute `process_shape()` on each tuple in parallel.
        # the 'with' statement automatically calls the close() and join() methods on the pool
        with multiprocessing.Pool() as pool:
            self.shapes = pool.starmap_async(self.process_shape,
                                             shape_args).get()
        # check that detection box is expected value (27 for testing purposes)
        rolls = {
            "piece": {
                "bubbleValues":
                ["p", "b", "k", "q", "n", "r", "0-0", "-0",
                 "X"],  # placeholder for empty list
                "direction": "horizontal",
            },
            "pos_x": {
                "bubbleValues": ["a", "b", "c", "d", "e", "f", "g", "h", "+"],
                "direction": "horizontal",
            },
            "pos_y": {
                "bubbleValues": ["1", "2", "3", "4", "5", "6", "7", "8", "="],
                "direction": "horizontal",
            }
        }
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        # for shape in self.shapes:
        #     if len(shape.detection_boxes) == 27:
        #         for idx, detection_box in enumerate(shape.detection_boxes):
        #             #TODO Implement the DrawConfigs properly for transparent_box case
        #             src_image = detection_box.draw(
        #                 src_image,
        #                 label_shape=True,
        #                 label=str(idx),
        #                 draw_label_config=DrawConfigs.DEFAULT_LABEL,
        #                 draw_line_config=DrawConfigs.DEFAULT_LINE,
        #                 transparent_box=False,
        #                 display_image=False)
            
        for shape in self.shapes:
            if len(shape.detection_boxes) == 27:
                for idx, detection_box in enumerate(shape.detection_boxes):
                    if detection_box.isMarked:
                        detection_box.value = shape.detection_boxes.labels[idx]
                        #TODO Implement the DrawConfigs properly for transparent_box case
                        src_image = detection_box.draw(
                            src_image,
                            label_shape=True,
                            label=detection_box.value,
                            draw_label_config=DrawConfigs.DRAW_ON_TRANSPARENT,
                            draw_line_config=DrawConfigs.DRAW_TRANSPARENT_BOX,
                            transparent_box=True,
                            display_image=False)
            else:
                detection_box.value = ""
                        
        # TODO: At this point, we have all detection blocks and isMarked.
        # Now we need to group the detection boxes based on an inputted
        # template.json. Each group should have a corresponding FieldType
        # with the associated values. In our case, we know that each group
        # will have the same FieldType though that can't be guaranteed for
        # every case. Therefore, the only way to know is to define each in
        # the template.json.
        plshow("img", src_image)
        return src_image
        # Store the results in each shape object
        # for idx, shape in enumerate(self.shapes):
        #     shape.processing_mask = results[idx][0]
        #     shape.detection_boxes = results[idx][1]
        #     shape.detection_scores = results[idx][2]
        #     shape.process_detection_boxes()

    def process_shape(
        self,
        idx,
        shape,
        src_image,
        proc_template_method,
        box_detection_method,
        debug_level,
    ):
        """Use with multiprocessing - process a shape by extracting a sub-image, applying template processing, detecting boxes, sorting boxes, and returning the processed shape.

        Args:
            idx (int): The index of the shape.
            shape (Shape): The shape object to be processed.
            src_image (numpy.ndarray): The source image containing the shape.
            proc_template_method (str): The method for template processing.
            box_detection_method (str): The method for box detection.
            debug_level (int): The level of debug information to display.

        Returns:
            Shape: The processed shape object.
            
        Note:
            For use with multiprocessing.
        """
        proc_img = np.zeros_like(src_image)
        mask = np.zeros_like(src_image)
        cv2.fillPoly(mask, [shape.contour], (255, 255, 255))
        sub_image = cv2.bitwise_and(src_image, mask)
        proc_template = self._get_proc_template(
            sub_image,
            idx,
            detection_method=proc_template_method,
            display_image=False,
            debug_level=debug_level)

        shape.detection_boxes = self._get_detection_boxes(
            sub_image,
            idx,
            detection_method=box_detection_method,
            display_image=False,
            debug_level=0)
        
        # with CodeTimer(f"Sort Shapes @ {idx}"):
        shape.detection_boxes = shape.detection_boxes.sort_shapes(
                    sort_method=ShapeSortMethods.LEFT_TO_RIGHT_TOP_TO_BOTTOM,
                    tolerance=50)
        # proc_img is flattened answer template. not filled in boxes marked boxes on source img
        proc_sub_img = proc_template.draw(
            proc_img,
            draw_line_config=DrawConfigs.DRAW_ON_BLANK,
            display_image=False)
        # if we invert proc_img, the marked boxes are now the filled boxes
        inverted_proc_sub_img = 255 - proc_sub_img
        shape.processing_mask = inverted_proc_sub_img
        shape.process_detection_boxes()
        isMarked = shape.isMarked

        return shape

    def process(
        self,
        src_image,
        proc_template_method=DetectionMethod.METHOD_6,
        box_detection_method=DetectionMethod.METHOD_7,
        debug_level=0,
    ):
        """Process a shape sequentially.
            Args:
                src_image (numpy.ndarray): The source image to be processed.
                proc_template_method (int, optional): The method used for processing templates.
                    Defaults to DetectionMethod.METHOD_6.
                box_detection_method (int, optional): The method used for detecting boxes.
                    Defaults to DetectionMethod.METHOD_7.
                debug_level (int, optional): The debug level for controlling verbosity.
                    Defaults to 0.

            Returns:
                numpy.ndarray: The processed image with marked detection boxes.

            Note:
                The `src_image` is modified in-place.
        """

        # TODO Add in validation checks based on template...eg. check shapes match rows*cols

        if debug_level == 1:
            blank_image = np.zeros_like(src_image)
            blank_image.fill(255)
            blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
            img_detection_boxes = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        proc_img = np.zeros_like(src_image)
        # For each parent shape
        for idx, shape in enumerate(self.shapes):
            mask = np.zeros_like(src_image)
            cv2.fillPoly(mask, [shape.contour], (255, 255, 255))
            sub_image = cv2.bitwise_and(src_image, mask)
            proc_template = self._get_proc_template(
                sub_image,
                idx,
                detection_method=proc_template_method,
                display_image=False,
                debug_level=debug_level,
            )
            shape.detection_boxes = self._get_detection_boxes(
                sub_image,
                idx,
                detection_method=box_detection_method,
                display_image=False,
                debug_level=0,
            )

            test_boxes = shape.detection_boxes
            start_time = time.perf_counter()
            shape.detection_boxes = shape.detection_boxes.sort_shapes(
                sort_method=ShapeSortMethods.LEFT_TO_RIGHT_TOP_TO_BOTTOM,
                tolerance=50)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print("Execution time for Sort Orig: {:.6f} seconds".format(
                execution_time))
            # proc_img is flattened answer template. not filled in boxes marked boxes on source img
            proc_sub_img = proc_template.draw(
                proc_img,
                draw_line_config=DrawConfigs.DRAW_ON_BLANK,
                display_image=False,
            )
            # if we invert proc_img, the marked boxes are now the filled boxes
            proc_sub_img = 255 - proc_img
            shape.processing_mask = proc_sub_img
            shape.process_detection_boxes()
            if shape.detection_boxes:
                for idx, detection_box in enumerate(shape.detection_boxes):
                    detection_box.draw(
                        src_image,
                        label_shape=True,
                        label=str(idx),
                        draw_label_config=DrawConfigs.DEFAULT_LABEL,
                        draw_line_config=DrawConfigs.DEFAULT_LINE,
                        display_image=False,
                    )
        plshow("src", src_image)
        return src_image
        # plshow("Final", src_image)
        if debug_level == 1:
            # High contrast helps to identify detection box irregularies
            detection_boxes.draw(
                blank_image,
                label_shapes=True,
                draw_label_config=DrawConfigs.DEFAULT_LINE,
                draw_line_config=DrawConfigs.DEFAULT_LABEL,
                display_image=False,
            )
            # Shows detection boxes drawn on src image
            detection_boxes.draw(
                img_detection_boxes,
                label_shapes=True,
                draw_label_config=DrawConfigs.DEFAULT_LINE,
                draw_line_config=DrawConfigs.DEFAULT_LABEL,
                display_image=False,
            )
            plshow("Detected Gridlines (proc_img)", proc_img)
            plshow("Detected Gridlines (Blank)", blank_image)
            plshow("Detection Boxes (Overlayed on Color)", img_detection_boxes)

    def _get_proc_template(
        self,
        sub_image,
        idx,
        detection_method=DetectionMethod.METHOD_6,
        debug_level=0,
        display_image=False,
    ):
        proc_template = None
        if detection_method == DetectionMethod.METHOD_6:
            # We need to get the proc_template and for checking subimages
            proc_template, pt_contours = gridfinder.method6(
                src_image=sub_image, idx=idx, debug_level=debug_level)
            proc_template = ShapeArray(proc_template)
            if display_image and proc_template is not None:
                proc_template.draw(
                    image=sub_image,
                    label_shapes=False,
                    draw_label_config=DrawConfigs.DRAW_ON_BLANK,
                )
                plshow(f"_get_proc_template({idx}, {detection_method})",
                       sub_image)
            return proc_template
        elif detection_method == DetectionMethod.METHOD_1:
            NotImplemented()
        else:
            raise ValueError("Invalid detection method")

    def _get_detection_boxes(
        self,
        sub_image,
        idx,
        detection_method=DetectionMethod.METHOD_7,
        debug_level=0,
        display_image=False,
    ):
        if detection_method == DetectionMethod.METHOD_7:
            detection_boxes, db_contours = gridfinder.method7(
                sub_image, idx, debug_level)
            # this is convenient but should sorting happen earlier?
            detection_boxes = ShapeArray(detection_boxes)
        if display_image:
            # Draws detection box on a blank sub_image - not helpful yet
            detection_boxes.draw(
                sub_image,
                label_shapes=False,
                draw_label_config=DrawConfigs.DRAW_ON_BLANK,
                draw_line_config=DrawConfigs.DRAW_ON_BLANK,
                display_image=False,
            )
            plshow(f"_get_detection_boxes({idx}, {detection_method})",
                   sub_image)
        return detection_boxes

    def get_sub_shapes(self):
        for idx, shape in enumerate(self.shapes):
            continue

def remove_contours(image, contours):
    for c in contours:
        cv2.drawContours(image, [c], -1, (0, 0, 0), -1)
    return image
def imgsize(sub_image):
    # Get the size of the sub-image in bytes
    sub_image_size_bytes = sub_image.nbytes

    # Convert bytes to kilobytes (KB) and megabytes (MB)
    sub_image_size_kb = sub_image_size_bytes / 1024
    sub_image_size_mb = sub_image_size_kb / 1024
    # Print the sub-image size in KB and MB
    print("Sub-image size: {} KB ({:.2f} MB)".format(int(sub_image_size_kb),
                                                     sub_image_size_mb))
def sort_points_clockwise(points):
    center = np.mean(points, axis=0)  # Calculate the center point
    angles = np.arctan2(
        points[:, 1] - center[1], points[:, 0] -
        center[0])  # Calculate angles relative to the center point
    sorted_indices = np.argsort(angles)  # Sort indices based on angles
    sorted_points = points[
        sorted_indices]  # Sort the points based on the sorted indices
    return sorted_points
def plshow(title, image):
    plt.title(title)
    plt.imshow(image)
    plt.show()