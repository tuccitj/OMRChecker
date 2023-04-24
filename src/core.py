from itertools import groupby
import math
import os
from collections import defaultdict
from pprint import pprint
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

import src.constants as constants
from src.logger import logger
from src.utils.image import CLAHE_HELPER, ImageUtils
from src.utils.interaction import InteractionUtils
from PIL import Image

class ImageInstanceOps:
    """Class to hold fine-tuned utilities for a group of images. One instance for each processing directory."""
    #TODO check for flag in entry.py line 267 ifAutoGen and Type == rectangle and if so run, generate_field_blocks, then generate_custom_labels, if possible, auto gen bubble dimensions...possibly based on role #
    save_img_list: Any = defaultdict(list)

    def __init__(self, tuning_config):
        super().__init__()
        self.tuning_config = tuning_config
        self.save_image_level = tuning_config.outputs.save_image_level

    def apply_preprocessors(self, file_path, in_omr, template):
        tuning_config = self.tuning_config
        # resize to conform to template
        in_omr = ImageUtils.resize_util(
            in_omr,
            tuning_config.dimensions.processing_width,
            tuning_config.dimensions.processing_height,
        )

        # run pre_processors in sequence
        for pre_processor in template.pre_processors:
            in_omr = pre_processor.apply_filter(in_omr, file_path)
        return in_omr
    
    def generate_template(self, in_omr, template):
        field_blocks = []
        tuning_config = self.tuning_config
        # resize to conform to template
        in_omr = ImageUtils.resize_util(
            in_omr,
            tuning_config.dimensions.processing_width,
            tuning_config.dimensions.processing_height,
        )
        boxes,img2 = self.get_rectangle_coords(in_omr)
        # test = self.get_test(in_omr)
        
        # Determine Bubble Dimensions [width, height]
        x, y, w, h = boxes[0]
        width = w//9
        height = h//3
        template.bubble_dimensions = [width, height]
        template.bubbles_gap = width
        sub_rect_coords = []
        for x, y, w, h in rect_coords:
            bubble_height = template.bubble_dimensions[1]
            sub_rect_coords.extend([(x, y + i*bubble_height, w, bubble_height) for i in range(3)])
        if len(template.field_blocks_object) == len(sub_rect_coords):
            mapped_field_blocks = {}
            for (x, y, w, _), (key, field_block) in zip(sub_rect_coords, template.field_blocks_object.items()):
                field_block["origin"] = [x,y]
                field_block["bubblesGap"] = w / 9
                # mapped_field_blocks[key] = field_block
            # template.field_blocks_object = mapped_field_blocks
        else:
            print("Too many or too few rectangles detected")
            
        # for key, coord in enumerate(rect_coords):
        #     x,y,w,h = coord
        #     field_block = {}
        #     field_block[f"m{key}_w"] = "fieldType": s 
        return template
        # we need to get the orientation and the number of elements in a roll...this is all in constants.py - how to access? we also do need the template
        # so here is a design decision - if we are to generate based on field_type, the field types would  need to be provided in the template...if not,
        # specify in the template.json options...
    
    def get_test(self, in_omr):
        image = in_omr
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

        # Iterate through the contours and filter for polygons
        polygons = []
        for cnt in contours:
            # Approximate the contour as a polygon
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            # Draw the polygon on the original image
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
            # Only consider polygons with 4 vertices
        
    
        plt.imshow(image)
        plt.show()
    # ONLY IF PREPROCESSORS HAS BEEN RUN AND ONLY IF OMR_MARKER USED TO CROP PAGE
    def get_rectangle_coords(self, in_omr):
        image = in_omr
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # # Convert the image to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #       # Apply Gaussian blur to reduce noise
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Convert the image to grayscale
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # Convert the image to grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # # Determine optimal threshold using Otsu's method
        # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Apply thresholding with a threshold value of 127
        ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)   
        # Apply edge detection to the grayscale image
        edged = cv2.Canny(thresh, 10, 200)
        
        # Find contours in the edge map
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Initialize a list to store the coordinates of the rectangles
        boxes = []
        # Iterate through the contours and filter for rectangles
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        for cnt in contours:
            # approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            # Assume 'contour' is a list of points representing a polygon
            # rect = cv2.minAreaRect(cnt)
            # boxPoints = cv2.boxPoints(rect)
            # box = np.int0(boxPoints)
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    
            _, _, w, h = cv2.boundingRect(approx)
            if w > 200 and h > 200 and w < 1000 and h < 1000:  # Only consider rectangles with a width and height greater than 50 pixels
                vertices = approx.ravel().reshape(-1, 2)
                # Draw the vertices on the image
                for vertex in vertices:
                    cv2.circle(image, tuple(vertex), 5, (0, 0, 255), -1)
                # cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)
                box =  np.int0([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])    
                box = self.sort_vertices(box)  
                box = np.int0(box) 
                boxes.append(box)
                # box = self.sort_by_position(approx)
                # box = np.int0(box)
                # boxes.append(box)
                # box = self.sort_by_position(box)
                # box = np.int0(box)
                # boxes.append(box)
            # polygons = boxes
            # for polygon in polygons:
            #     # Sort the vertices of the polygon by their x-coordinate
            #     sorted_indices = np.argsort(polygon[:, 0])
            #     sorted_polygon = polygon[sorted_indices]

            #     # Create empty lists to hold the vertices of the three resulting polygons
            #     left_polygon = []
            #     middle_polygon = []
            #     right_polygon = []

               
            #     # Reverse the order of the vertices in the right polygon
            #     right_polygon.reverse()
            #     left_polygon= np.int0(left_polygon)
            #     middle_polygon= np.int0(middle_polygon)
            #     right_polygon= np.int0(right_polygon)
                
            #     cv2.polylines(image, [left_polygon], True, (0, 255, 0), 2)
            #     cv2.polylines(image, [middle_polygon], True, (0, 255, 0), 2)
            #     cv2.polylines(image, [right_polygon], True, (0, 255, 0), 2)
                
            #     plt.imshow(image)
            #     plt.show()

        sorted_boxes = sorted(boxes, key=lambda box: box[0][0])
        
        ##TODO tolerance should either be determined mathmatically w/distributions or set as a constant in config
        tolerance = 30      
        # group coordinates by x value with tolerance and sort each group by y value
        rect_coords_grouped = []
        for key, group in groupby(sorted_boxes, lambda x: round(x[0][0]/tolerance)):
            rect_coords_grouped.extend(sorted(list(group), key=lambda x: x[0][1]))
        img = np.zeros((4000, 3000, 1), dtype=np.uint8)
        # Draw the boxes and label them
        for i, box in enumerate(rect_coords_grouped):
            print(boxes[i-1])
            print(i, box)
            # cv2.drawContours(image, [box[0]], 0, (255, 255, 255), 2) 
            # Draw the rectangle on the image
            cv2.polylines(image, [box], True, (0, 0, 255), 2)
            label = "{}".format(i+1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (box[0][0], box[0][1]- 5), font, 5, (0, 0, 255), 5)
            # cv2.fillPoly(img, [box], 255)
            height = np.linalg.norm(box[3][1]-box[0][1])
            split_height = int(height//3)
            #don't keep x the same...find y, scope left until boundary is hit - that is the new uppermost left
            upper_left = box[0]
            upper_right = box[1]
            lower_right = [box[1][0], box[1][1] + split_height]
            lower_left = [box[0][0], box[0][1] + split_height]
     
            top_third = np.array([upper_left,upper_right,lower_right, lower_left])
            upper_left = lower_left
            upper_right = lower_right
            lower_right = [box[1][0], box[1][1] + 2*split_height]
            lower_left = [box[0][0], box[0][1] + 2*split_height]
            middle_third = np.array([upper_left,upper_right,lower_right, lower_left])
            
            upper_left = lower_left
            upper_right = lower_right
            lower_right = [box[1][0], box[1][1] + 3*split_height]
            lower_left = [box[0][0], box[0][1] + 3*split_height]
            bottom_third = np.array([upper_left,upper_right,lower_right, lower_left])

            # top_third=[upper_left,upper_right,lower_right, lower_left]
            # cv2.drawContours(image, [top_third], 0, (255, 255, 255), 2)
            # cv2.drawContours(image, [middle_third], 0, (255, 255, 255), 2)
            # cv2.drawContours(image, [bottom_third], 0, (255, 255, 255), 2)
            

           
            # # calculate the slope of the top line
            # topLine = [box[0], box[1]]
            # middleline = [[box[0][0]-25,box[0][1]+split_height],[box[1][0]+25,box[1][1]+split_height]]
            # bottom = [[box[0][0],box[0][1]+2*split_height],[box[1][0],box[1][1]+2*split_height]]
            # # Create a mask of the same size as the image
            # mask = np.zeros_like(img)            
            # # Draw the top half of the polygon on the mask
            # cv2.drawContours(mask, [box], 0, 255, -1)
            # cv2.rectangle(mask, (0,0), (mask.shape[1],middleline[0][1]), 0, -1)
            # # Split the polygon based on the line
            # split_mask = np.zeros_like(img)
            # cv2.rectangle(split_mask, middleline[0], middleline[1], 255, -1)
            # top_half = cv2.bitwise_and(mask, split_mask)
            # bottom_half = cv2.bitwise_and(mask, cv2.bitwise_not(split_mask))
            # # Display the results
            # plt.title("Original")
            # plt.imshow(img)
            # plt.show()
            # plt.title("Top Half")
            # plt.imshow(top_half)
            # plt.show()
            # plt.title("Bottom Half")
            # plt.imshow(bottom_half)
            # plt.show()
            #_______________________--
            # cv2.line(img, topLine[0], topLine[1], (255, 255, 255), 2)
            # cv2.line(img, middleline[0], middleline[1], (255, 255, 255), 2)
            # cv2.line(img, bottom[0], bottom[1], (255, 255, 255), 2)
            # Split the polygon vertically
            split_point = box[0][1] + height//2
            pts1 = box[np.where(box[:, 1] <= split_point)]
            pts2 = box[np.where(box[:, 1] > split_point)]

            # Draw the polygons
            cv2.drawContours(img, [pts1], 0, 1, -1)
            cv2.drawContours(img, [pts2], 0, 1, -1)

            M = cv2.moments(box)
            print(M)
            # Calculate the centroid
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # # # Draw the centroid on the image
            # # cv2.circle(image, (cx, cy), 5, (0, 0, 255), 100)
            # Draw label at the centroid of the box
            label = "{}".format(i+1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (cx, cy), font, 3, (0, 0, 255), 3)
            image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
            
        # # DEMONSTRATION Draw the rectangles on the image in the sorted order
        # for i, (x, y, w, h) in enumerate(rect_coords_grouped):
        #     # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     label = "{}".format(i+1)
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(image, label, (x, y - 5), font, 5, (0, 0, 255), 5)
        #     image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)

        plt.imshow(image)
        plt.show()
        
        return rect_coords_grouped, image
    
    def sort_vertices(self, box):
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
            return(box)
    
    def gen_field_blocks():
        #naming convention
        #
        return("")
    
    def gen_custom_labels():
        return("")
    
    def read_omr_response(self, template, image, name, save_dir=None):
        config = self.tuning_config
        auto_align = config.alignment_params.auto_align
        try:
            img = image.copy()
            # origDim = img.shape[:2]
            img = ImageUtils.resize_util(
                img, template.page_dimensions[0], template.page_dimensions[1]
            )
            if img.max() > img.min():
                img = ImageUtils.normalize_util(img)
            # Processing copies
            transp_layer = img.copy()
            final_marked = img.copy()

            morph = img.copy()
            self.append_save_img(3, morph)

            if auto_align:
                # Note: clahe is good for morphology, bad for thresholding
                morph = CLAHE_HELPER.apply(morph)
                self.append_save_img(3, morph)
                # Remove shadows further, make columns/boxes darker (less gamma)
                morph = ImageUtils.adjust_gamma(
                    morph, config.threshold_params.GAMMA_LOW
                )
                # TODO: all numbers should come from either constants or config
                _, morph = cv2.threshold(morph, 220, 220, cv2.THRESH_TRUNC)
                morph = ImageUtils.normalize_util(morph)
                self.append_save_img(3, morph)
                if config.outputs.show_image_level >= 4:
                    InteractionUtils.show("morph1", morph, 0, 1, config)

            # Move them to data class if needed
            # Overlay Transparencies
            alpha = 0.65
            box_w, box_h = template.bubble_dimensions
            omr_response = {}
            multi_marked, multi_roll = 0, 0

            # TODO Make this part useful for visualizing status checks
            # blackVals=[0]
            # whiteVals=[255]

            if config.outputs.show_image_level >= 5:
                all_c_box_vals = {"int": [], "mcq": []}
                # TODO: simplify this logic
                q_nums = {"int": [], "mcq": []}

            # Find Shifts for the field_blocks --> Before calculating threshold!
            if auto_align:
                # print("Begin Alignment")
                # Open : erode then dilate
                v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
                morph_v = cv2.morphologyEx(
                    morph, cv2.MORPH_OPEN, v_kernel, iterations=3
                )
                _, morph_v = cv2.threshold(morph_v, 200, 200, cv2.THRESH_TRUNC)
                morph_v = 255 - ImageUtils.normalize_util(morph_v)

                if config.outputs.show_image_level >= 3:
                    InteractionUtils.show(
                        "morphed_vertical", morph_v, 0, 1, config=config
                    )

                # InteractionUtils.show("morph1",morph,0,1,config=config)
                # InteractionUtils.show("morphed_vertical",morph_v,0,1,config=config)

                self.append_save_img(3, morph_v)

                morph_thr = 60  # for Mobile images, 40 for scanned Images
                _, morph_v = cv2.threshold(morph_v, morph_thr, 255, cv2.THRESH_BINARY)
                # kernel best tuned to 5x5 now
                morph_v = cv2.erode(morph_v, np.ones((5, 5), np.uint8), iterations=2)

                self.append_save_img(3, morph_v)
                # h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
                # morph_h = cv2.morphologyEx(morph, cv2.MORPH_OPEN, h_kernel, iterations=3)
                # ret, morph_h = cv2.threshold(morph_h,200,200,cv2.THRESH_TRUNC)
                # morph_h = 255 - normalize_util(morph_h)
                # InteractionUtils.show("morph_h",morph_h,0,1,config=config)
                # _, morph_h = cv2.threshold(morph_h,morph_thr,255,cv2.THRESH_BINARY)
                # morph_h = cv2.erode(morph_h,  np.ones((5,5),np.uint8), iterations = 2)
                if config.outputs.show_image_level >= 3:
                    InteractionUtils.show(
                        "morph_thr_eroded", morph_v, 0, 1, config=config
                    )

                self.append_save_img(6, morph_v)

                # template relative alignment code
                for field_block in template.field_blocks:
                    s, d = field_block.origin, field_block.dimensions

                    match_col, max_steps, align_stride, thk = map(
                        config.alignment_params.get,
                        [
                            "match_col",
                            "max_steps",
                            "stride",
                            "thickness",
                        ],
                    )
                    shift, steps = 0, 0
                    while steps < max_steps:
                        left_mean = np.mean(
                            morph_v[
                                s[1] : s[1] + d[1],
                                s[0] + shift - thk : -thk + s[0] + shift + match_col,
                            ]
                        )
                        right_mean = np.mean(
                            morph_v[
                                s[1] : s[1] + d[1],
                                s[0]
                                + shift
                                - match_col
                                + d[0]
                                + thk : thk
                                + s[0]
                                + shift
                                + d[0],
                            ]
                        )

                        # For demonstration purposes-
                        if(field_block.name == "int1"):
                            ret = morph_v.copy()
                            cv2.rectangle(ret,
                                          (s[0]+shift-thk,s[1]),
                                          (s[0]+shift+thk+d[0],s[1]+d[1]),
                                          constants.CLR_WHITE,
                                          3)
                            appendSaveImg(6,ret)
                        print(shift, left_mean, right_mean)
                        left_shift, right_shift = left_mean > 100, right_mean > 100
                        if left_shift:
                            if right_shift:
                                break
                            else:
                                shift -= align_stride
                        else:
                            if right_shift:
                                shift += align_stride
                            else:
                                break
                        steps += 1

                    field_block.shift = shift
                    print("Aligned field_block: ",field_block.name,"Corrected Shift:",
                      field_block.shift,", dimensions:", field_block.dimensions,
                      "origin:", field_block.origin,'\n')
                print("End Alignment")

            final_align = None
            if config.outputs.show_image_level >= 2:
                initial_align = self.draw_template_layout(img, template, shifted=False)
                final_align = self.draw_template_layout(
                    img, template, shifted=True, draw_qvals=True
                )
                # appendSaveImg(4,mean_vals)
                self.append_save_img(2, initial_align)
                self.append_save_img(2, final_align)

                if auto_align:
                    final_align = np.hstack((initial_align, final_align))
                    print("FINAL ALIGN", final_align)
            self.append_save_img(5, img)

            # Get mean bubbleValues n other stats
            all_q_vals, all_q_strip_arrs, all_q_std_vals = [], [], []
            total_q_strip_no = 0
            for field_block in template.field_blocks:
                q_std_vals = []
                for field_block_bubbles in field_block.traverse_bubbles:
                    q_strip_vals = []
                    for pt in field_block_bubbles:
                        # shifted
                        x, y = (pt.x + field_block.shift, pt.y)
                        rect = [y, y + box_h, x, x + box_w]
                        q_strip_vals.append(
                            cv2.mean(img[rect[0] : rect[1], rect[2] : rect[3]])[0]
                            # detectCross(img, rect) ? 100 : 0
                        )
                    q_std_vals.append(round(np.std(q_strip_vals), 2))
                    all_q_strip_arrs.append(q_strip_vals)
                    # _, _, _ = get_global_threshold(q_strip_vals, "QStrip Plot",
                    #   plot_show=False, sort_in_plot=True)
                    # hist = getPlotImg()
                    # InteractionUtils.show("QStrip "+field_block_bubbles[0].field_label, hist, 0, 1,config=config)
                    all_q_vals.extend(q_strip_vals)
                    # print(total_q_strip_no, field_block_bubbles[0].field_label, q_std_vals[len(q_std_vals)-1])
                    total_q_strip_no += 1
                all_q_std_vals.extend(q_std_vals)
            global_std_thresh, _, _ = self.get_global_threshold(
                all_q_std_vals
            )  # , "Q-wise Std-dev Plot", plot_show=True, sort_in_plot=True)
            # plt.show()
            # hist = getPlotImg()
            # InteractionUtils.show("StdHist", hist, 0, 1,config=config)

            # Note: Plotting takes Significant times here --> Change Plotting args
            # to support show_image_level
            # , "Mean Intensity Histogram",plot_show=True, sort_in_plot=True)
            global_thr, _, _ = self.get_global_threshold(all_q_vals, looseness=4)

            logger.info(
                f"Thresholding:\tglobal_thr: {round(global_thr, 2)} \tglobal_std_THR: {round(global_std_thresh, 2)}\t{'(Looks like a Xeroxed OMR)' if (global_thr == 255) else ''}"
            )
            # plt.show()
            # hist = getPlotImg()
            # InteractionUtils.show("StdHist", hist, 0, 1,config=config)

            # if(config.outputs.show_image_level>=1):
            #     hist = getPlotImg()
            #     InteractionUtils.show("Hist", hist, 0, 1,config=config)
            #     appendSaveImg(4,hist)
            #     appendSaveImg(5,hist)
            #     appendSaveImg(2,hist)

            per_omr_threshold_avg, total_q_strip_no, total_q_box_no = 0, 0, 0
            for field_block in template.field_blocks:
                block_q_strip_no = 1
                shift = field_block.shift
                s, d = field_block.origin, field_block.dimensions
                key = field_block.name[:3]
                # cv2.rectangle(final_marked,(s[0]+shift,s[1]),(s[0]+shift+d[0],
                #   s[1]+d[1]),CLR_BLACK,3)
                for field_block_bubbles in field_block.traverse_bubbles:
                    # All Black or All White case
                    no_outliers = all_q_std_vals[total_q_strip_no] < global_std_thresh
                    # print(total_q_strip_no, field_block_bubbles[0].field_label,
                    #   all_q_std_vals[total_q_strip_no], "no_outliers:", no_outliers)
                    per_q_strip_threshold = self.get_local_threshold(
                        all_q_strip_arrs[total_q_strip_no],
                        global_thr,
                        no_outliers,
                        f"Mean Intensity Histogram for {key}.{field_block_bubbles[0].field_label}.{block_q_strip_no}",
                        config.outputs.show_image_level >= 6,
                    )
                    # print(field_block_bubbles[0].field_label,key,block_q_strip_no, "THR: ",
                    #   round(per_q_strip_threshold,2))
                    per_omr_threshold_avg += per_q_strip_threshold

                    # Note: Little debugging visualization - view the particular Qstrip
                    # if(
                    #     0
                    #     # or "q17" in (field_block_bubbles[0].field_label)
                    #     # or (field_block_bubbles[0].field_label+str(block_q_strip_no))=="q15"
                    #  ):
                    #     st, end = qStrip
                    #     InteractionUtils.show("QStrip: "+key+"-"+str(block_q_strip_no),
                    #     img[st[1] : end[1], st[0]+shift : end[0]+shift],0,config=config)

                    # TODO: get rid of total_q_box_no
                    detected_bubbles = []
                    for bubble in field_block_bubbles:
                        bubble_is_marked = (
                            per_q_strip_threshold > all_q_vals[total_q_box_no]
                        )
                        total_q_box_no += 1
                        if bubble_is_marked:
                            detected_bubbles.append(bubble)
                            x, y, field_value = (
                                bubble.x + field_block.shift,
                                bubble.y,
                                bubble.field_value,
                            )
                            cv2.rectangle(
                                final_marked,
                                (int(x + box_w / 12), int(y + box_h / 12)),
                                (
                                    int(x + box_w - box_w / 12),
                                    int(y + box_h - box_h / 12),
                                ),
                                constants.CLR_DARK_GRAY,
                                3,
                            )

                            cv2.putText(
                                final_marked,
                                str(field_value),
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                constants.TEXT_SIZE,
                                (20, 20, 10),
                                int(1 + 3.5 * constants.TEXT_SIZE),
                            )
                        else:
                            cv2.rectangle(
                                final_marked,
                                (int(x + box_w / 10), int(y + box_h / 10)),
                                (
                                    int(x + box_w - box_w / 10),
                                    int(y + box_h - box_h / 10),
                                ),
                                constants.CLR_GRAY,
                                -1,
                            )

                    for bubble in detected_bubbles:
                        field_label, field_value = (
                            bubble.field_label,
                            bubble.field_value,
                        )
                        # Only send rolls multi-marked in the directory
                        multi_marked_local = field_label in omr_response
                        omr_response[field_label] = (
                            (omr_response[field_label] + field_value)
                            if multi_marked_local
                            else field_value
                        )
                        # TODO: generalize this into identifier
                        # multi_roll = multi_marked_local and "Roll" in str(q)
                        multi_marked = multi_marked or multi_marked_local

                    if len(detected_bubbles) == 0:
                        field_label = field_block_bubbles[0].field_label
                        omr_response[field_label] = field_block.empty_val

                    if config.outputs.show_image_level >= 5:
                        if key in all_c_box_vals:
                            q_nums[key].append(f"{key[:2]}_c{str(block_q_strip_no)}")
                            all_c_box_vals[key].append(
                                all_q_strip_arrs[total_q_strip_no]
                            )

                    block_q_strip_no += 1
                    total_q_strip_no += 1
                # /for field_block

            per_omr_threshold_avg /= total_q_strip_no
            per_omr_threshold_avg = round(per_omr_threshold_avg, 2)
            # Translucent
            cv2.addWeighted(
                final_marked, alpha, transp_layer, 1 - alpha, 0, final_marked
            )
            # Box types
            if config.outputs.show_image_level >= 5:
                # plt.draw()
                f, axes = plt.subplots(len(all_c_box_vals), sharey=True)
                f.canvas.manager.set_window_title(name)
                ctr = 0
                type_name = {
                    "int": "Integer",
                    "mcq": "MCQ",
                    "med": "MED",
                    "rol": "Roll",
                }
                for k, boxvals in all_c_box_vals.items():
                    axes[ctr].title.set_text(type_name[k] + " Type")
                    axes[ctr].boxplot(boxvals)
                    # thrline=axes[ctr].axhline(per_omr_threshold_avg,color='red',ls='--')
                    # thrline.set_label("Average THR")
                    axes[ctr].set_ylabel("Intensity")
                    axes[ctr].set_xticklabels(q_nums[k])
                    # axes[ctr].legend()
                    ctr += 1
                # imshow will do the waiting
                plt.tight_layout(pad=0.5)
                plt.show()

            if config.outputs.show_image_level >= 3 and final_align is not None:
                final_align = ImageUtils.resize_util_h(
                    final_align, int(config.dimensions.display_height)
                )
                # [final_align.shape[1],0])
                InteractionUtils.show(
                    "Template Alignment Adjustment", final_align, 0, 0, config=config
                )

            if config.outputs.save_detections and save_dir is not None:
                if multi_roll:
                    save_dir = save_dir.joinpath("_MULTI_")
                image_path = str(save_dir.joinpath(name))
                ImageUtils.save_img(image_path, final_marked)

            self.append_save_img(2, final_marked)

            if save_dir is not None:
                for i in range(config.outputs.save_image_level):
                    self.save_image_stacks(i + 1, name, save_dir)

            return omr_response, final_marked, multi_marked, multi_roll

        except Exception as e:
            raise e
        
    @staticmethod
    def draw_template_layout(img, template, shifted=True, draw_qvals=False, border=-1):
        img = ImageUtils.resize_util(
            img, template.page_dimensions[0], template.page_dimensions[1]
        )
        final_align = img.copy()
        box_w, box_h = template.bubble_dimensions
        for field_block in template.field_blocks:
            s, d = field_block.origin, field_block.dimensions
            shift = field_block.shift
            if shifted:
                cv2.rectangle(
                    final_align,
                    (s[0] + shift, s[1]),
                    (s[0] + shift + d[0], s[1] + d[1]),
                    constants.CLR_BLACK,
                    3,
                )
            else:
                cv2.rectangle(
                    final_align,
                    (s[0], s[1]),
                    (s[0] + d[0], s[1] + d[1]),
                    constants.CLR_BLACK,
                    3,
                )
            for field_block_bubbles in field_block.traverse_bubbles:
                for pt in field_block_bubbles:
                    x, y = (pt.x + field_block.shift, pt.y) if shifted else (pt.x, pt.y)
                    cv2.rectangle(
                        final_align,
                        (int(x + box_w / 10), int(y + box_h / 10)),
                        (int(x + box_w - box_w / 10), int(y + box_h - box_h / 10)),
                        constants.CLR_GRAY,
                        border,
                    )
                    if draw_qvals:
                        rect = [y, y + box_h, x, x + box_w]
                        cv2.putText(
                            final_align,
                            f"{int(cv2.mean(img[rect[0] : rect[1], rect[2] : rect[3]])[0])}",
                            (rect[2] + 2, rect[0] + (box_h * 2) // 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            constants.CLR_BLACK,
                            2,
                        )
            if shifted:
                text_in_px = cv2.getTextSize(
                    field_block.name, cv2.FONT_HERSHEY_SIMPLEX, constants.TEXT_SIZE, 4
                )
                cv2.putText(
                    final_align,
                    field_block.name,
                    (int(s[0] + d[0] - text_in_px[0][0]), int(s[1] - text_in_px[0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    constants.TEXT_SIZE,
                    constants.CLR_BLACK,
                    4,
                )
        return final_align

    def get_global_threshold(
        self,
        q_vals_orig,
        plot_title=None,
        plot_show=True,
        sort_in_plot=True,
        looseness=1,
    ):
        """
        Note: Cannot assume qStrip has only-gray or only-white bg
            (in which case there is only one jump).
        So there will be either 1 or 2 jumps.
        1 Jump :
                ......
                ||||||
                ||||||  <-- risky THR
                ||||||  <-- safe THR
            ....||||||
            ||||||||||

        2 Jumps :
                ......
                |||||| <-- wrong THR
            ....||||||
            |||||||||| <-- safe THR
            ..||||||||||
            ||||||||||||

        The abstract "First LARGE GAP" is perfect for this.
        Current code is considering ONLY TOP 2 jumps(>= MIN_GAP) to be big,
            gives the smaller one

        """
        config = self.tuning_config
        PAGE_TYPE_FOR_THRESHOLD, MIN_JUMP, JUMP_DELTA = map(
            config.threshold_params.get,
            [
                "PAGE_TYPE_FOR_THRESHOLD",
                "MIN_JUMP",
                "JUMP_DELTA",
            ],
        )

        global_default_threshold = (
            constants.GLOBAL_PAGE_THRESHOLD_WHITE
            if PAGE_TYPE_FOR_THRESHOLD == "white"
            else constants.GLOBAL_PAGE_THRESHOLD_BLACK
        )

        # Sort the Q bubbleValues
        # TODO: Change var name of q_vals
        q_vals = sorted(q_vals_orig)
        # Find the FIRST LARGE GAP and set it as threshold:
        ls = (looseness + 1) // 2
        l = len(q_vals) - ls
        max1, thr1 = MIN_JUMP, global_default_threshold
        for i in range(ls, l):
            jump = q_vals[i + ls] - q_vals[i - ls]
            if jump > max1:
                max1 = jump
                thr1 = q_vals[i - ls] + jump / 2

        # NOTE: thr2 is deprecated, thus is JUMP_DELTA
        # Make use of the fact that the JUMP_DELTA(Vertical gap ofc) between
        # values at detected jumps would be atleast 20
        max2, thr2 = MIN_JUMP, global_default_threshold
        # Requires atleast 1 gray box to be present (Roll field will ensure this)
        for i in range(ls, l):
            jump = q_vals[i + ls] - q_vals[i - ls]
            new_thr = q_vals[i - ls] + jump / 2
            if jump > max2 and abs(thr1 - new_thr) > JUMP_DELTA:
                max2 = jump
                thr2 = new_thr
        # global_thr = min(thr1,thr2)
        global_thr, j_low, j_high = thr1, thr1 - max1 // 2, thr1 + max1 // 2

        # # For normal images
        # thresholdRead =  116
        # if(thr1 > thr2 and thr2 > thresholdRead):
        #     print("Note: taking safer thr line.")
        #     global_thr, j_low, j_high = thr2, thr2 - max2//2, thr2 + max2//2

        if plot_title:
            _, ax = plt.subplots()
            ax.bar(range(len(q_vals_orig)), q_vals if sort_in_plot else q_vals_orig)
            ax.set_title(plot_title)
            thrline = ax.axhline(global_thr, color="green", ls="--", linewidth=5)
            thrline.set_label("Global Threshold")
            thrline = ax.axhline(thr2, color="red", ls=":", linewidth=3)
            thrline.set_label("THR2 Line")
            # thrline=ax.axhline(j_low,color='red',ls='-.', linewidth=3)
            # thrline=ax.axhline(j_high,color='red',ls='-.', linewidth=3)
            # thrline.set_label("Boundary Line")
            # ax.set_ylabel("Mean Intensity")
            ax.set_ylabel("Values")
            ax.set_xlabel("Position")
            ax.legend()
            if plot_show:
                plt.title(plot_title)
                plt.show()

        return global_thr, j_low, j_high

    def get_local_threshold(
        self, q_vals, global_thr, no_outliers, plot_title=None, plot_show=True
    ):
        """
        TODO: Update this documentation too-
        //No more - Assumption : Colwise background color is uniformly gray or white,
                but not alternating. In this case there is atmost one jump.

        0 Jump :
                        <-- safe THR?
            .......
            ...|||||||
            ||||||||||  <-- safe THR?
        // How to decide given range is above or below gray?
            -> global q_vals shall absolutely help here. Just run same function
                on total q_vals instead of colwise _//
        How to decide it is this case of 0 jumps

        1 Jump :
                ......
                ||||||
                ||||||  <-- risky THR
                ||||||  <-- safe THR
            ....||||||
            ||||||||||

        """
        config = self.tuning_config
        # Sort the Q bubbleValues
        q_vals = sorted(q_vals)

        # Small no of pts cases:
        # base case: 1 or 2 pts
        if len(q_vals) < 3:
            thr1 = (
                global_thr
                if np.max(q_vals) - np.min(q_vals) < config.threshold_params.MIN_GAP
                else np.mean(q_vals)
            )
        else:
            # qmin, qmax, qmean, qstd = round(np.min(q_vals),2), round(np.max(q_vals),2),
            #   round(np.mean(q_vals),2), round(np.std(q_vals),2)
            # GVals = [round(abs(q-qmean),2) for q in q_vals]
            # gmean, gstd = round(np.mean(GVals),2), round(np.std(GVals),2)
            # # DISCRETION: Pretty critical factor in reading response
            # # Doesn't work well for small number of values.
            # DISCRETION = 2.7 # 2.59 was closest hit, 3.0 is too far
            # L2MaxGap = round(max([abs(g-gmean) for g in GVals]),2)
            # if(L2MaxGap > DISCRETION*gstd):
            #     no_outliers = False

            # # ^Stackoverflow method
            # print(field_label, no_outliers,"qstd",round(np.std(q_vals),2), "gstd", gstd,
            #   "Gaps in gvals",sorted([round(abs(g-gmean),2) for g in GVals],reverse=True),
            #   '\t',round(DISCRETION*gstd,2), L2MaxGap)

            # else:
            # Find the LARGEST GAP and set it as threshold: //(FIRST LARGE GAP)
            l = len(q_vals) - 1
            max1, thr1 = config.threshold_params.MIN_JUMP, 255
            for i in range(1, l):
                jump = q_vals[i + 1] - q_vals[i - 1]
                if jump > max1:
                    max1 = jump
                    thr1 = q_vals[i - 1] + jump / 2
            # print(field_label,q_vals,max1)

            confident_jump = (
                config.threshold_params.MIN_JUMP
                + config.threshold_params.CONFIDENT_SURPLUS
            )
            # If not confident, then only take help of global_thr
            if max1 < confident_jump:
                if no_outliers:
                    # All Black or All White case
                    thr1 = global_thr
                else:
                    # TODO: Low confidence parameters here
                    pass

            # if(thr1 == 255):
            #     print("Warning: threshold is unexpectedly 255! (Outlier Delta issue?)",plot_title)

        # Make a common plot function to show local and global thresholds
        if plot_show and plot_title is not None:
            _, ax = plt.subplots()
            ax.bar(range(len(q_vals)), q_vals)
            thrline = ax.axhline(thr1, color="green", ls=("-."), linewidth=3)
            thrline.set_label("Local Threshold")
            thrline = ax.axhline(global_thr, color="red", ls=":", linewidth=5)
            thrline.set_label("Global Threshold")
            ax.set_title(plot_title)
            ax.set_ylabel("Bubble Mean Intensity")
            ax.set_xlabel("Bubble Number(sorted)")
            ax.legend()
            # TODO append QStrip to this plot-
            # appendSaveImg(6,getPlotImg())
            if plot_show:
                plt.show()
        return thr1

    def append_save_img(self, key, img):
        if self.save_image_level >= int(key):
            self.save_img_list[key].append(img.copy())

    def save_image_stacks(self, key, filename, save_dir):
        config = self.tuning_config
        if self.save_image_level >= int(key) and self.save_img_list[key] != []:
            name = os.path.splitext(filename)[0]
            result = np.hstack(
                tuple(
                    [
                        ImageUtils.resize_util_h(img, config.dimensions.display_height)
                        for img in self.save_img_list[key]
                    ]
                )
            )
            result = ImageUtils.resize_util(
                result,
                min(
                    len(self.save_img_list[key]) * config.dimensions.display_width // 3,
                    int(config.dimensions.display_width * 2.5),
                ),
            )
            ImageUtils.save_img(f"{save_dir}stack/{name}_{str(key)}_stack.jpg", result)

    def reset_all_save_img(self):
        for i in range(self.save_image_level):
            self.save_img_list[i + 1] = []
