from itertools import groupby
import cv2
from matplotlib import pyplot as plt
import numpy as np
import src.custom.gridfinder as gridfinder
class DrawConfig:
    def __init__(self, fontFace, fontScale, color, thickness):
        self.fontFace = fontFace
        self.fontScale = fontScale
        self.color = color
        self.thickness = thickness
        
class DrawConfigs:
    DEFAULT = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
    DEFAULT_LINE = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
    DEFAULT_LABEL = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
    UPPER_LEFT_LARGE_LABEL = DrawConfig(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(0,0,255), thickness=5)
 
class DetectionMethod:
    METHOD_1 = 1
    METHOD_2 = 2
    METHOD_3 = 3
    METHOD_4 = 4  
    METHOD_5 = 5   
class ShapeSortMethods:
    DEFAULT = 0
    LEFT_TO_RIGHT_TOP_TO_BOTTOM = 1
    TOP_TO_BOTTOM_LEFT_TO_RIGHT = 2
    
    
          
    
class Shape:
    def __init__(self, contour, draw_line_config=DrawConfigs.DEFAULT_LINE, draw_label_config=DrawConfigs.DEFAULT_LABEL):
        self.contour = contour
        self.vertices = self._get_vertices(contour, sort=True)
        self.draw_line_config = draw_line_config
        self.draw_label_config = draw_label_config
    
    def draw(self, image, label_shape=False, label='', draw_line_config='', draw_label_config='', display_image=False):
        # Set DrawConfigs to defaults
        if not draw_line_config:
            draw_line_config = self.draw_line_config
        if not draw_label_config:
            draw_label_config = self.draw_label_config
        # draw the shape
        cv2.polylines(image, [self.contour], True, draw_line_config.color, draw_line_config.thickness)
        # identify the label position
        self.label_position = (self.vertices[0][0], self.vertices[0][1]-5)
        if label_shape:
            self.label(image, label, self.draw_label_config)
        if display_image:
            plshow("shape", image)
        return image
    def label(self, image, label, draw_config):
        if draw_config:
            cv2.putText(image, label, self.label_position, draw_config.fontFace, draw_config.fontScale, draw_config.color, draw_config.thickness)
        else:
            raise ValueError('No draw configuration provided')
        return image  
    def _get_vertices(self, contour, sort):
        cnt = contour
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        vertices =  np.int0([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
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
        sorted_lowest_y_list = sorted(lowest_y_list, key=lambda point: point[0])
        # Sort highest_y_list by the highest x value to the lowest x value
        sorted_highest_y_list = sorted(highest_y_list, key=lambda point: point[0], reverse=True)
        sorted_lowest_y_list.extend(sorted_highest_y_list)
        box = sorted_lowest_y_list.copy()
        box = np.int0(box)
        return(box)

class ShapeArray:
    def __init__(self, shapes, sort_method=ShapeSortMethods.DEFAULT):
        self.shapes = shapes
        self.shapes = self._sort_shapes(shapes, sort_method)
    
    def __getitem__(self, index):
        return self.shapes[index]
    def __len__(self):
        return len(self.shapes)
    def _sort_shapes(self, shapes, sort_method, tolerance=30):
        shapes_final = []
        if sort_method == ShapeSortMethods.TOP_TO_BOTTOM_LEFT_TO_RIGHT:
            sorted_shapes = sorted(shapes, key=lambda shape: shape.vertices[0][0])
            ##TODO tolerance should either be determined mathmatically w/distributions or set as a constant in config
            tolerance = tolerance      
            # group coordinates by x value with tolerance and sort each group by y value
            for key, group in groupby(sorted_shapes, lambda shape: round(shape.vertices[0][0]/tolerance)):
                shapes_final.extend(sorted(list(group), key=lambda shape: shape.vertices[0][1]))
        elif sort_method == ShapeSortMethods.LEFT_TO_RIGHT_TOP_TO_BOTTOM:
            ## TODO tolerance should be a parameter...should be able to set defaults somewhere
            tolerance = 50      
            sorted_shapes = sorted(shapes, key=lambda shape: shape.vertices[0][1])
            ##TODO tolerance should either be determined mathmatically w/distributions or set as a constant in config
            # group coordinates by x value with tolerance and sort each group by y value
            for key, group in groupby(sorted_shapes, lambda shape: round(shape.vertices[0][1]/tolerance)):
                shapes_final.extend(sorted(list(group), key=lambda shape: shape.vertices[0][0]))
        return shapes_final
    
    def draw(self, image, label_shapes=False, draw_line_config="", draw_label_config="", display_image=False):
        for idx, shape in enumerate(self.shapes):
            label = "{}".format(idx+1)
            if draw_line_config:
                shape.draw_line_config=draw_line_config
            if draw_label_config:
                shape.draw_label_config=draw_label_config
            shape.draw(image, label_shape=label_shapes, label = label)
            # if label_shapes:
            #     label = "{}".format(idx+1)
            #     shape.draw(image, label_shapes=True, label = label)
            #     shape.label(image, label, font)
        if display_image:
            plshow('ShapeArray', image)
        return image
    
    def detect_gridlines(self, src_image, detection_method=DetectionMethod.METHOD_1, display_image=False):
        blank_image = np.zeros_like(src_image)
        blank_image.fill(255)
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
        disp_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        
        for idx, shape in enumerate(self.shapes):
            mask = np.zeros_like(src_image)
            cv2.drawContours(mask, [shape.contour], 0, 255, -1)
            # Apply the mask to the original image
            sub_image = cv2.bitwise_and(src_image, mask)
            if detection_method == DetectionMethod.METHOD_1:
                sub_shapes = gridfinder.method1(sub_image, idx)
                sub_shapes = ShapeArray(sub_shapes)
                sub_shapes.draw(blank_image, label_shapes=True)
            elif detection_method == DetectionMethod.METHOD_2:
                sub_shapes = gridfinder.method2(sub_image, idx)
                sub_shapes = ShapeArray(sub_shapes)
                sub_shapes.draw(blank_image, label_shapes=True)
            elif detection_method == DetectionMethod.METHOD_4:
                sub_shapes = gridfinder.method4(sub_image, idx)
                sub_shapes = ShapeArray(sub_shapes)
                sub_shapes.draw(blank_image, label_shapes=True)
            elif detection_method == DetectionMethod.METHOD_5:
                sub_shapes, contours = gridfinder.method7(sub_image, idx, debug_level=0)
                cv2.drawContours(disp_img, contours, -1, (0,255,0), -1) 
                sub_shapes = ShapeArray(sub_shapes, sort_method=ShapeSortMethods.LEFT_TO_RIGHT_TOP_TO_BOTTOM)
                sub_shapes.draw(blank_image, label_shapes=True)
            else:
                raise ValueError('Invalid detection method')
        if display_image:
            plshow("Detected Gridlines (Blank)", blank_image)
            plshow("Detected Gridlines (Image)", disp_img)
                  

    
    def get_sub_shapes(self):
        for idx, shape in enumerate(self.shapes):
            continue
        
def remove_contours(image, contours):
    for c in contours:
        cv2.drawContours(image, [c], -1, (0,0,0), -1)
    return image
def plshow(title, image):
    plt.title(title)
    plt.imshow(image)
    plt.show()
def g_lines(src_image, thresh):
    filter=True
    lines = cv2.HoughLines(thresh,1,np.pi/180,150)
    if not lines.any():
        print('No lines were found')
        exit()

    if filter:
        rho_threshold = 15
        theta_threshold = 0.1

        # how many lines are similar to a given one
        similar_lines = {i : [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i,theta_i = lines[i][0]
                rho_j,theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x : len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines)*[True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]: # and only if we have not disregarded them already
                    continue

                rho_i,theta_i = lines[indices[i]][0]
                rho_j,theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

    print('number of Hough lines:', len(lines))

    filtered_lines = []

    if filter:
        for i in range(len(lines)): # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

        print('Number of filtered lines:', len(filtered_lines))
    else:
        filtered_lines = lines

    for line in filtered_lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(src_image,(x1,y1),(x2,y2),(0,255,0),2)

    plshow("g_lines", src_image)