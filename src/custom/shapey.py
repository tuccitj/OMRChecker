import multiprocessing
import time
from itertools import groupby
import cv2
import numpy as np
from matplotlib import pyplot as plt
import src.custom.gridfinder as gridfinder
from src.logger import logger


def tsimeit(method):

    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        print(f"{method.__name__}: {end_time - start_time}s")
        return result

    return timed


class DrawConfig:

    def __init__(self, fontFace, fontScale, color, thickness):
        self.fontFace = fontFace
        self.fontScale = fontScale
        self.color = color
        self.thickness = thickness


class DrawConfigs:
    """Repository of configurations for use with cv2 functions such as DrawContours(), PolyLines(), and PutText()"""

    DEFAULT = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale=1,
                         color=(0, 0, 255),
                         thickness=2)
    DEFAULT_LINE = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=1,
                              color=(0, 0, 255),
                              thickness=2)
    DEFAULT_LABEL = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=1,
                               color=(0, 0, 255),
                               thickness=2)
    UPPER_LEFT_LARGE_LABEL = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=5,
                                        color=(0, 0, 255),
                                        thickness=5)
    IMG_PROC_TEMPLATE = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                   fontScale=1,
                                   color=(0, 0, 0),
                                   thickness=-1)
    DRAW_ON_BLANK = DrawConfig(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=-1,
    )


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


class Shape:
    """
    Smallest unit for processing. A shape is usually within a ShapeArray
    which represents a Field Block in the template.
    """

    def __init__(
        self,
        contour,
        draw_line_config=DrawConfigs.DEFAULT_LINE,
        draw_label_config=DrawConfigs.DEFAULT_LABEL,
    ):
        self.contour = contour
        self.vertices = self._get_vertices(contour, sort=True)
        self.draw_line_config = draw_line_config
        self.draw_label_config = draw_label_config
        self.isMarked = False
        self.detection_boxes = None
        self.processing_mask = None

    def draw(
        self,
        image,
        label_shape=False,
        label="",
        draw_line_config=None,
        draw_label_config=None,
        display_image=False,
    ):
        # Set DrawConfigs to defaults
        if not draw_line_config:
            draw_line_config = self.draw_line_config
        if not draw_label_config:
            draw_label_config = self.draw_label_config
        # draw the shape
        # cv2.polylines(image, [self.contour], True, draw_line_config.color, draw_line_config.thickness)
        cv2.drawContours(
            image,
            [self.contour],
            -1,
            draw_line_config.color,
            draw_line_config.thickness,
        )
        # identify the label position
        self.label_position = (self.vertices[0][0], self.vertices[0][1] - 5)
        if label_shape:
            self.label(image, label, self.draw_label_config)
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

    # def multi_process_detection_boxes(self):
    #     with multiprocessing.Pool(5) as p:
    #         results = p.starmap(self._process_detection_box, enumerate(self.detection_boxes))
    #         p.close()
    #         p.join()

    #         for idx, detection_box in enumerate(self.detection_boxes):
    #             detection_box.isMarked = results[idx]

    # def _process_detection_box(self, idx, detection_box):
    #     return detection_box._meetsBlackThreshold(sub_image=self.processing_mask, debug_level=0)

    def process_detection_boxes(self):
        if self.detection_boxes:
            for idx, detection_box in enumerate(self.detection_boxes):
                detection_box.isMarked = detection_box._meetsBlackThreshold(
                    sub_image=self.processing_mask, debug_level=0)

    # def process_detection_boxes(self):
    #     # Submit each detection box to the process pool as a separate task
    #     tasks = []
    #     for idx, detection_box in enumerate(self.detection_boxes):
    #         task = pool.submit_task(detection_box._meetsBlackThreshold, (self.processing_mask, 0))
    #         tasks.append(task)

    #     # Wait for all tasks to complete and retrieve the results
    #     for idx, task in enumerate(tasks):
    #         self.detection_boxes[idx].isMarked = pool.get_task_result(task)
    # for idx, detection_box in enumerate(self.detection_boxes):
    #     detection_box.isMarked = detection_box._meetsBlackThreshold(sub_image=self.processing_mask, debug_level=0)
    def _get_vertices(self, contour, sort):
        cnt = contour
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        vertices = np.int0(
            [approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
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


class ShapeArray:
    """Array of Shapes"""

    def __init__(self, shapes, sort_method=None):
        self.shapes = shapes
        if sort_method:
            self.shapes = self._sort_shapes(shapes, sort_method)

    def __getitem__(self, index):
        return self.shapes[index]

    def __len__(self):
        return len(self.shapes)

    def _sort_shapes(self, shapes, sort_method, tolerance=30):
        """Method for sorting shapes in a FieldBlock

        Args:
            shapes (_type_): _description_
            sort_method (_type_): Sort method to sort by
            tolerance (int, optional): _description_. Defaults to 30.

        Returns:
            _type_: _description_
        """
        shapes_final = []
        if sort_method == ShapeSortMethods.TOP_TO_BOTTOM_LEFT_TO_RIGHT:
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
            # TODO tolerance should be a parameter...should be able to set defaults somewhere
            tolerance = 50
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
        # Generate a list of (index, shape) tuples to be processed by `process_shape()`.
        shape_args = [(
            idx,
            shape,
            src_image,
            proc_template_method,
            box_detection_method,
            debug_level,
        ) for idx, shape in enumerate(self.shapes)]

        # Use `map_async()` to execute `process_shape()` on each tuple in parallel.
        with multiprocessing.Pool() as pool:
            results = pool.starmap_async(self.process_shape, shape_args).get()
        # Close the pool to release its resources.
        pool.join()
        pool.close()
        for shape in results:
            for detection_box in shape.detection_boxes:
                if detection_box.isMarked:
                    detection_box.draw(
                        src_image,
                        label_shape=True,
                        label=str(detection_box.isMarked),
                        draw_label_config=DrawConfigs.DEFAULT_LABEL,
                        draw_line_config=DrawConfigs.DEFAULT_LINE,
                        display_image=False,
                    )
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
        proc_img = np.zeros_like(src_image)
        mask = np.zeros_like(src_image)
        cv2.fillPoly(mask, [shape.contour], (255, 255, 255))
        sub_image = cv2.bitwise_and(src_image, mask)
        proc_template = self._get_proc_template(
            sub_image,
            idx,
            detection_method=proc_template_method,
            display_image=False,
            debug_level=debug_level,
            show_image=False,
        )
        shape.detection_boxes = self._get_detection_boxes(
            sub_image,
            idx,
            detection_method=box_detection_method,
            display_image=False,
            debug_level=debug_level,
        )
        # proc_img is flattened answer template. not filled in boxes marked boxes on source img
        if proc_template:
            proc_sub_img = proc_template.draw(
                proc_img,
                draw_line_config=DrawConfigs.DRAW_ON_BLANK,
                display_image=False)
        # if we invert proc_img, the marked boxes are now the filled boxes
        proc_sub_img = 255 - proc_img
        shape.processing_mask = proc_sub_img
        shape.process_detection_boxes()
        return shape

    def process(
        self,
        src_image,
        proc_template_method=DetectionMethod.METHOD_6,
        box_detection_method=DetectionMethod.METHOD_7,
        debug_level=0,
    ):
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
                show_image=False,
            )
            shape.detection_boxes = self._get_detection_boxes(
                sub_image,
                idx,
                detection_method=box_detection_method,
                display_image=False,
                debug_level=debug_level,
            )
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
                for detection_box in shape.detection_boxes:
                    if detection_box.isMarked:
                        detection_box.draw(
                            src_image,
                            label_shape=True,
                            label=str(detection_box.isMarked),
                            draw_label_config=DrawConfigs.DEFAULT_LABEL,
                            draw_line_config=DrawConfigs.DEFAULT_LINE,
                            display_image=False,
                        )

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
        display_image=False,
        debug_level=0,
        show_image=False,
    ):
        if detection_method == DetectionMethod.METHOD_6:
            # We need to get the proc_template and for checking subimages
            proc_template, pt_contours = gridfinder.method6(
                src_image=sub_image, idx=idx, debug_level=debug_level)
            proc_template = ShapeArray(proc_template)
            if show_image:
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
        display_image=False,
        debug_level=0,
        show_image=False,
    ):
        detection_boxes = None
        if detection_method == DetectionMethod.METHOD_7:
            detection_boxes, db_contours = gridfinder.method7(sub_image,
                                                              idx,
                                                              debug_level=0)
            # this is convenient but should sorting happen earlier?
            detection_boxes = ShapeArray(
                detection_boxes,
                sort_method=ShapeSortMethods.LEFT_TO_RIGHT_TOP_TO_BOTTOM,
            )
        if show_image and detection_boxes is not None:
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

    # def _isMarked(self, proc_sub_img, detection_boxes):
    #     for detection_box in detection_boxes:

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


def plshow(title, image):
    plt.title(title)
    plt.imshow(image)
    plt.show()