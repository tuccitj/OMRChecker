import itertools
import cv2
from imutils import contours as cntx
import numpy as np
from src.custom.shapey import DrawConfigs, Shape, ShapeArray, plshow
from itertools import chain


# https://stackoverflow.com/questions/59182827/how-to-get-the-cells-of-a-sudoku-grid-with-opencv
def method1(image, idx=""):
    show = False
    # Load image, grayscale, and adaptive threshold
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5
    )
    if show:
        plshow("thresh", thresh)
    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        # this is filtering out the letters and numbers - it doesnt filter out the filled in squares
        if area < 2000:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    v_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    if show:
        plshow("v_thresh", v_thresh)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    h_thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1
    )
    if show:
        plshow("h_thresh", h_thresh)

    img_bin_final = v_thresh | h_thresh
    if show:
        plshow("ibf", img_bin_final)
    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(
        img_bin_final,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )

    # Draw the detected lines on the input image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_bin_final, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if show:
        plshow("ibf", img_bin_final)
    cnts = cv2.findContours(img_bin_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        # this is filtering out the letters and numbers - it doesnt filter out the filled in squares
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
    if show:
        plshow("ibfin", img_bin_final)
    # plshow('tfin', thresh)
    # final_kernel = np.ones((3,3), np.uint8)
    # img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=1)

    # # Sort by top to bottom and each row by left to right
    # invert = 255 - thresh
    final_kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.dilate(thresh, final_kernel, iterations=5)

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    (cnts, _) = cntx.sort_contours(cnts, method="top-to-bottom")

    # cv2.drawContours(image, cnts, -1, (255,255,255), -1)
    shapes = []
    contours = []
    for i, cnt in enumerate(cnts):
        if i == 0:
            continue
        shape = Shape(cnt)
        shapes.append(shape)
        contours.append(cnt)
    shapes = ShapeArray(shapes)
    return shapes

    # color = (0, 0, 255 * (i / len(cnts)))
    # cv2.drawContours(image, [cnt], -1, color, 2)
    # M = cv2.moments(cnt)
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    # cv2.putText(image, f"{i+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # sudoku_rows = []
    # row = []
    # for (i, c) in enumerate(cnts, 1):
    #     area = cv2.contourArea(c)
    #     cv2.polylines(image, [c], True, (0, 0, 255), 2)
    #     if area < 50000:
    #         row.append(c)
    #         if i % 9 == 0:
    #             (cnts, _) = cntx.sort_contours(row, method="left-to-right")
    #             sudoku_rows.append(cnts)
    #             row = []

    # # Iterate through each box
    # for row in sudoku_rows:
    #     for c in row:
    #         mask = np.zeros(image.shape, dtype=np.uint8)
    #         cv2.drawContours(mask, [c], -1, (255,255,255), -1)
    #         result = cv2.bitwise_and(image, mask)
    #         result[mask==0] = 255
    #         cv2.imshow('result', result)
    #         cv2.waitKey(175)


# https://stackoverflow.com/questions/65717860/extract-boxes-from-sudoku-in-opencv/65732398#65732398
def method2(image, idx=""):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plshow("Gray_scale", gray_scale)
    ### Performing canny edge detection and adding a dilation layer
    img_bin = cv2.adaptiveThreshold(
        gray_scale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    # plshow("thresh", thresh)

    # img_bin = cv2.Canny(gray_scale,50,110)
    # plshow("img_bin canny", img_bin)
    dil_kernel = np.ones((3, 3), np.uint8)
    img_bin = cv2.dilate(img_bin, dil_kernel, iterations=1)
    # plshow("img_bin", img_bin)

    ### assuming minimum box size would be 20*20
    line_min_width = 50
    ### finding horizontal lines
    kernal_h = np.ones((1, line_min_width), np.uint8)
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
    # plshow("img_bin_h", img_bin_h)

    ### finding vertical lines
    kernal_v = np.ones((line_min_width, 1), np.uint8)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
    # plshow("img_bin_v", img_bin_v)

    ### merging and adding a dilation layer to close small gaps
    img_bin_final = img_bin_h | img_bin_v
    final_kernel = np.ones((3, 3), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)
    # plshow("img_bin_f", img_bin_final)

    ### applying connected component analysis
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ~img_bin_final, connectivity=8, ltype=cv2.CV_32S
    )
    ###drawing rectangles on the detected boxes.
    ### 1 and 0 and the background and residue connected components whihc we do not require
    shapes = []
    for x, y, w, h, area in stats[2:]:
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        if area > 1000:
            # cv2.putText(image,'box',(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,0), 2)
            # boxes.append([x, y, w, h])
            shape = Shape([x, y, w, h])
            shapes.append(shape)
            # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    # plshow("image", image)
    shapes = ShapeArray(shapes)
    return shapes


# https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python
def method3(image, idx=""):
    filter = True
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # kernel = np.ones((5,5),np.uint8)
    # edges = cv2.erode(edges,kernel,iterations = 1)
    cv2.imwrite("canny.jpg", edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    if not lines.any():
        print("No lines were found")
        exit()

    if filter:
        rho_threshold = 15
        theta_threshold = 0.1

        # how many lines are similar to a given one
        similar_lines = {i: [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i, theta_i = lines[i][0]
                rho_j, theta_j = lines[j][0]
                if (
                    abs(rho_i - rho_j) < rho_threshold
                    and abs(theta_i - theta_j) < theta_threshold
                ):
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x: len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines) * [True]
        for i in range(len(lines) - 1):
            if not line_flags[
                indices[i]
            ]:  # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(
                i + 1, len(lines)
            ):  # we are only considering those elements that had less similar line
                if not line_flags[
                    indices[j]
                ]:  # and only if we have not disregarded them already
                    continue

                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                if (
                    abs(rho_i - rho_j) < rho_threshold
                    and abs(theta_i - theta_j) < theta_threshold
                ):
                    line_flags[
                        indices[j]
                    ] = False  # if it is similar and have not been disregarded yet then drop it now

    print("number of Hough lines:", len(lines))

    filtered_lines = []

    if filter:
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

        print("Number of filtered lines:", len(filtered_lines))
    else:
        filtered_lines = lines

    for line in filtered_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite("hough.jpg", img)


def method4(image, idx=""):
    idx = idx + 1
    show = False
    if idx == 8 or idx == 46 or idx == 52:
        show = True
    show = False
    # Load image, grayscale, and adaptive threshold
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    if show:
        plshow("thresh", thresh)
    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        # this is filtering out the letters and numbers - it doesnt filter out the filled in squares
        if area < 2000:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
    # plshow("thresh", thresh)
    dil_kernel = np.ones((3, 3), np.uint8)
    img_bin = cv2.dilate(thresh, dil_kernel, iterations=2)

    if show:
        plshow("dilate", img_bin)
    ### assuming minimum box size would be 20*20
    line_min_width = 100
    ### finding horizontal lines
    kernal_h = np.ones((1, line_min_width), np.uint8)
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 10))
    # Perform morphological closing on the image with the horizontal kernel
    img_bin_h = cv2.dilate(img_bin_h, kernel, iterations=1)

    if show:
        plshow("img_bin_h", img_bin_h)

    ### finding vertical lines
    kernal_v = np.ones((line_min_width, 1), np.uint8)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 200))
    # Dilate the image using the vertical structuring element
    img_bin_v = cv2.dilate(img_bin_v, kernel, iterations=1)
    # img_bin_v = cv2.morphologyEx(img_bin_v, cv2.MORPH_CLOSE, kernel)
    if show:
        plshow("img_bin_v", img_bin_v)

    ### merging and adding a dilation layer to close small gaps
    img_bin_final = img_bin_h | img_bin_v
    final_kernel = np.ones((3, 3), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)
    if show:
        plshow("ibf", img_bin_final)
    cnts = cv2.findContours(img_bin_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    (cnts, _) = cntx.sort_contours(cnts, method="top-to-bottom")

    # cv2.drawContours(image, cnts, -1, (255,255,255), -1)
    contours = []
    for i, cnt in enumerate(cnts):
        if i == 0:
            continue
        contours.append(cnt)
        color = (0, 0, 255 * (i / len(cnts)))
        cv2.drawContours(image, [cnt], -1, color, 2)
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(image, f"{i}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # plshow("fin",image)
    return contours


def method5(image, idx="", debug_level=0):
    # Load image, grayscale, and adaptive threshold
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blank_image = np.zeros_like(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 100,)
    # Define kernel size and create kernel
    kernel_size = 50
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = image
    # Iterate over each pixel in the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Extract the sub-image/kernel centered around this pixel
            kernel_image = img[
                max(0, i - kernel_size // 2) : min(
                    img.shape[0], i + kernel_size // 2 + 1
                ),
                max(0, j - kernel_size // 2) : min(
                    img.shape[1], j + kernel_size // 2 + 1
                ),
            ]

            # Compute the sum of the pixel values in the kernel
            kernel_sum = np.sum(kernel_image)

            # If the sum of the pixel values in the kernel is greater than 50% of the total pixel intensity in the kernel
            if kernel_sum / (kernel_size * kernel_size) > 0.5:
                img[i, j] = 255  # set pixel to white
            else:
                img[i, j] = 0  # set pixel to black

    # Apply erode operation to resulting image
    eroded = cv2.erode(img, kernel)
    plshow("eroded", eroded)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 1
    )
    if debug_level == 1:
        plshow("thresh", thresh)
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        # this is filtering out the letters and numbers - it doesnt filter out the filled in squares
        if area < 2000:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
    if debug_level == 1:
        plshow("After Removing Most Noise", thresh)
    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    v_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    if debug_level == 1:
        plshow("v_thresh-fixed", v_thresh)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    h_thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1
    )
    if debug_level == 1:
        plshow("h_thresh-fixed", h_thresh)

    img_bin_final = v_thresh | h_thresh
    if debug_level == 1:
        plshow("combined post fix h and v", img_bin_final)
    plshow("imf", img_bin_final)
    # Define minimum and maximum aspect ratios for regular shapes
    final_kernel = np.ones((3, 3), np.uint8)
    img_bin_final = cv2.erode(img_bin_final, final_kernel, iterations=3)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=9)

    plshow("_dil", img_bin_final)

    cnts = cv2.findContours(img_bin_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    MIN_ASPECT_RATIO = 0.9
    MAX_ASPECT_RATIO = 1.1

    # Iterate over each contour and filter out irregular shapes
    filtered_contours = []
    for contour in cnts:
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area
        if solidity >= 0.8:
            filtered_contours.append(contour)
        # x, y, w, h = cv2.boundingRect(contour)
        # aspect_ratio = float(w) / h
        # if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
        #     filtered_contours.append(contour)

    # Draw the filtered contours on the image
    cv2.drawContours(blank_image, filtered_contours, -1, (0, 255, 0), -1)
    plshow("bi", blank_image)
    run_this = True
    if run_this:
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(
            img_bin_final,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )
        # Draw the detected lines on the input image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_bin_final, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if debug_level == 1 or idx + 1 in [7, 21, 22, 30]:
            plshow("detected hough lines", img_bin_final)
        cnts = cv2.findContours(img_bin_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            # this is drawing blank pixels wherever the contours lie
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
        if debug_level == 1 or idx + 1 in [7, 21, 22, 30]:
            plshow("Removed Detected Hough Lines", thresh)

    # # Sort by top to bottom and each row by left to right
    # invert = 255 - thresh
    final_kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, final_kernel, iterations=7)
    if debug_level == 1:
        plshow("Dilated Remaining Lines", thresh)
    final_kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, final_kernel, iterations=7)
    if debug_level == 1:
        plshow("Eroded Remaining Lines", thresh)

    # g_lines(image,thresh)

    # # Apply HoughLines transform
    # lines = cv2.HoughLines(thresh, rho=1, theta=np.pi/180, threshold=500)
    # horizontal_lines = []
    # angle_threshold = np.deg2rad(10)
    # # Draw lines on the image
    # for line in lines:
    #     rho, theta = line[0]
    #     if abs(theta - np.pi/2) < angle_threshold:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         x1 = int(x0 + 1000*(-b))
    #         y1 = int(y0 + 1000*(a))
    #         x2 = int(x0 - 1000*(-b))
    #         y2 = int(y0 - 1000*(a))
    #         horizontal_lines.append((y1, y2))
    #     # cv2.line(thresh, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # if debug_level == 2:
    #     plshow("Hough Lines", thresh)
    # # Apply Hough transform to detect lines
    # lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180, threshold=100, minLineLength=1, maxLineGap=100)

    # # Draw the detected lines on the input image
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(thresh, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # if debug_level == 1:
    #     plshow("Hough Lines", thresh)
    # # Define the minimum line length threshold
    # min_length = 50
    # # Apply Hough transform to detect horizontal lines
    # horizontal_lines = cv2.HoughLines(thresh, rho=1, theta=np.pi/180, threshold=100)
    # horizontal_filtered_lines = []
    # if horizontal_lines is not None:
    #     # Calculate the average length of horizontal lines detected
    #     total_length = 0
    #     num_lines = 0
    #     for line in horizontal_lines:
    #         rho, theta = line[0]
    #         if np.abs(theta - np.pi/2) < np.pi/6: # Only consider lines with theta close to 90 degrees
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a*rho
    #             y0 = b*rho
    #             x1 = int(x0 + 1000*(-b))
    #             y1 = int(y0 + 1000*(a))
    #             x2 = int(x0 - 1000*(-b))
    #             y2 = int(y0 - 1000*(a))
    #             length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    #             total_length += length
    #             num_lines += 1
    #     if num_lines > 0:
    #         avg_length = total_length / num_lines

    #         # Calculate the minimum length threshold based on the average length
    #         min_length = int(0.8 * avg_length)

    #         # Filter horizontal lines based on minimum length threshold
    #         for line in horizontal_lines:
    #             rho, theta = line[0]
    #             if np.abs(theta - np.pi/2) < np.pi/6: # Only consider lines with theta close to 90 degrees
    #                 a = np.cos(theta)
    #                 b = np.sin(theta)
    #                 x0 = a*rho
    #                 y0 = b*rho
    #                 x1 = int(x0 + 1000*(-b))
    #                 y1 = int(y0 + 1000*(a))
    #                 x2 = int(x0 - 1000*(-b))
    #                 y2 = int(y0 - 1000*(a))
    #                 length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    #                 if length >= min_length:
    #                     horizontal_filtered_lines.append(line)

    #         # Draw the filtered horizontal lines on the input image
    #         for line in horizontal_filtered_lines:
    #             rho, theta = line[0]
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a*rho
    #             y0 = b*rho
    #             x1 = int(x0 + 1000*(-b))
    #             y1 = int(y0 + 1000*(a))
    #             x2 = int(x0 - 1000*(-b))
    #             y2 = int(y0 - 1000*(a))
    #             cv2.line(thresh, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # if debug_level==1:
    #     plshow("Horzontal Hough Lines", thresh)
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = cntx.sort_contours(cnts, method="top-to-bottom")

    # # Calculate the average area of all contours
    # areas = [cv2.contourArea(c) for c in cnts]
    # avg_area = sum(areas) / len(areas)

    # # Define a threshold range based on the average area
    # min_area = min(areas)
    # max_area = avg_area + 1000

    # # Filter out csontours whose areas are outside the threshold range
    # filtered_cnts = []
    # for c in cnts:
    #     area = cv2.contourArea(c)
    #     if area > 1500:
    #         filtered_cnts.append(c)

    #     # if area >= min_area and area <= max_area:
    #     #     filtered_cnts.append(c)
    blank_image = np.zeros_like(image)
    cv2.drawContours(blank_image, cnts, -1, (0, 255, 0), -1)
    cv2.drawContours(image, cnts, -1, (0, 255, 0), -1)
    if idx + 1 in [7, 21, 22, 30]:
        plshow(f"{idx} fin", image)
        plshow(f"{idx} blankfin", blank_image)

    # plshow("fadf", image)
    shapes = []
    contours = []
    for i, cnt in enumerate(cnts):
        if i == 0:
            continue
        shape = Shape(cnt)
        shapes.append(shape)
        contours.append(cnt)
    return shapes, contours
    # color = (0, 0, 255 * (i / len(cnts)))
    # cv2.drawContours(image, [cnt], -1, color, 2)
    # M = cv2.moments(cnt)
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    # cv2.putText(image, f"{i+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # sudoku_rows = []
    # row = []
    # for (i, c) in enumerate(cnts, 1):
    #     area = cv2.contourArea(c)
    #     cv2.polylines(image, [c], True, (0, 0, 255), 2)
    #     if area < 50000:
    #         row.append(c)
    #         if i % 9 == 0:
    #             (cnts, _) = cntx.sort_contours(row, method="left-to-right")
    #             sudoku_rows.append(cnts)
    #             row = []

    # # Iterate through each box
    # for row in sudoku_rows:
    #     for c in row:
    #         mask = np.zeros(image.shape, dtype=np.uint8)
    #         cv2.drawContours(mask, [c], -1, (255,255,255), -1)
    #         result = cv2.bitwise_and(image, mask)
    #         result[mask==0] = 255
    #         cv2.imshow('result', result)
    #         cv2.waitKey(175)


def method6_dep(image, idx="", debug_level=0):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5
    )
    if debug_level == 1:
        plshow("thresh", thresh)

    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2000:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
    if debug_level == 1:
        plshow("thres1", thresh)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)
    if debug_level == 1:
        plshow("thres2", thresh)
    mask = cv2.bitwise_and(image, image, mask=thresh)
    # plshow("mask", mask)

    # Sort by top to bottom and each row by left to right
    invert = 255 - thresh

    plshow("thresh", thresh)
    plshow("inverted", invert)
    # Apply bitwise AND between the inverted binary image and the original source image
    inverted_mask = cv2.bitwise_and(image, image, mask=invert)
    # plshow("inverted mask", inverted_mask)

    # mask = np.zeros_like(image)
    # cv2.drawContours(mask, [shape.contour], 0, 255, -1)
    # Apply the mask to the original image
    # sub_image = cv2.bitwise_and(src_image, mask)
    # plshow("invert", invert)
    norm = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    inv = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    norm = norm[0] if len(norm) == 2 else norm[1]
    inv = inv[0] if len(inv) == 2 else inv[1]
    cv2.drawContours(image, norm, -1, (0, 0, 255), -1)
    plshow("fdsaf", image)
    # cv2.drawContours(image, [inv], -1, (0,0,255), -1)

    # for i,cnt in enumerate(norm):
    # cv2.drawContours(merged, [cnt], -1, (0,0,255), -1)
    # # Compute the center of the contour using its moments
    # M = cv2.moments(cnt)
    # cx = int(M["m10"] / M["m00"])
    # cy = int(M["m01"] / M["m00"])
    # # Add a text label to the image with the contour's index
    # cv2.putText(merged, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Draw the merged contours on a blank image
    merged = np.zeros_like(image)
    im1 = np.zeros_like(image)
    nm2 = np.zeros_like(image)

    # cv2.drawContours(merged, norm, -1, (0,255,0), -1)
    for i, cnt in enumerate(norm):
        cv2.drawContours(merged, [cnt], -1, (0, 255, 0), 5)
        cv2.drawContours(nm2, [cnt], -1, (0, 255, 0), 5)

        # Compute the center of the contour using its moments
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Add a text label to the image with the contour's index
        cv2.putText(
            merged, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
        )
        cv2.putText(nm2, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    for i, cnt in enumerate(inv):
        cv2.drawContours(im1, [cnt], -1, (0, 0, 255), 2)
        cv2.drawContours(merged, [cnt], -1, (0, 0, 255), 2)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Add a text label to the image with the contour's index
        cv2.putText(
            merged, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(im1, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    plshow("im1", im1)
    plshow("nm2", nm2)
    plshow("merged", merged)

    # cv2.drawContours(merged, cnts, -1, (0, 0, 0), thickness=cv2.FILLED)
    # cv2.drawContours(merged, cnts2, -1, (0, 0, 0), thickness=cv2.FILLED)

    (cnts, _) = cntx.sort_contours(cnts, method="top-to-bottom")
    contours = []
    shapes = []
    for i, cnt in enumerate(cnts):
        if i == 0:
            continue
        shape = Shape(cnt)
        shapes.append(shape)
        contours.append(cnt)
    return shapes, contours


def method6(src_image, idx="", debug_level=0):
    gray = src_image.copy()
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5
    )
    if debug_level == 1:
        plshow("thresh", thresh)
    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2000:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
    if debug_level == 1:
        plshow("thres1", thresh)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)
    if debug_level == 1:
        plshow("thres2", thresh)

    invert = 255 - thresh

    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if debug_level == 1:
        cv2.drawContours(src_image, cnts, -1, (0, 0, 255), -1)
        plshow("fdsaf", src_image)
    shapes = []
    contours = []
    # Convert the tuple to a list
    cnts = list(cnts)
    # Remove the largest area contour
    largest_contour = max(cnts, key=cv2.contourArea)
    cnts.remove(largest_contour)
    for cnt in cnts:
        shape = Shape(cnt)
        shapes.append(shape)
        contours.append(cnt)

    return shapes, contours

    inv = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    norm = norm[0] if len(norm) == 2 else norm[1]
    inv = inv[0] if len(inv) == 2 else inv[1]

# for detection boxes
def method7(src_image, idx="", debug_level=0):
    if debug_level == 3:
        if idx + 1 in [46,48, 49, 50]:
            debug_level = 1
    
    img = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    gray = src_image.copy()
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    if debug_level == 1:
        plshow("Edges", edges)
    # Find horizontal and vertical lines using HoughLinesP
    edges = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 7
    )
    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2000:
            cv2.drawContours(edges, [c], -1, (0, 0, 0), -1)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
    )

    filtered_lines = []
    ## TODO Possibly...if expected # of shapes not detected, change this number
    angle_threshold = 0.3  # in radians
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
        if angle < angle_threshold or np.abs(angle - np.pi / 2) < angle_threshold:
            filtered_lines.append(line)

    # for line in filtered_lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Create empty binary images for horizontal and vertical lines
    h_lines = np.zeros_like(gray)
    v_lines = np.zeros_like(gray)
    # Find the length of the longest line
    longest_length = 0
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > longest_length:
            longest_length = length

    # Extend all lines to the length of the longest line
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if abs(y2 - y1) < abs(x2 - x1):
            # Horizontal line
            # Calculate the middle point of the line
            mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Extend the line to the length of the longest line
            x1_extended = int(mid_point[0] - length / 2)
            x2_extended = int(mid_point[0] + length / 2)
            y1_extended = y1
            y2_extended = y2

            cv2.line(
                h_lines,
                (x1_extended, y1_extended),
                (x2_extended, y2_extended),
                255,
                thickness=2,
            )
        else:
            # Vertical line
            # Calculate the middle point of the line
            mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Extend the line to the length of the longest line
            y1_extended = int(mid_point[1] - length / 2)
            y2_extended = int(mid_point[1] + length / 2)
            x1_extended = x1
            x2_extended = x2

            cv2.line(
                v_lines,
                (x1_extended, y1_extended),
                (x2_extended, y2_extended),
                255,
                thickness=2,
            )

    # Draw horizontal and vertical lines on the binary images
    # for line in filtered_lines:
    #     x1, y1, x2, y2 = line[0]
    #     if abs(y2 - y1) < abs(x2 - x1):
    #         # Horizontal line
    #         cv2.line(h_lines, (x1, y1), (x2, y2), 255, thickness=2)
    #     else:
    #         # Vertical line
    #         cv2.line(v_lines, (x1, y1), (x2, y2), 255, thickness=2)
    if debug_level == 1:
        plshow("h_lines", h_lines)
        plshow("v-lines", v_lines)
    # Dilate the horizontal and vertical lines binary images
    kernel_size = 5

    h_lines_dilated = cv2.dilate(
        h_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
        iterations=5,
    )
    v_lines_dilated = cv2.dilate(
        v_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
        iterations=5,
    )

    if debug_level == 1:
        plshow("h_lines_dilated", h_lines_dilated)
        plshow("v_lines_dilated", v_lines_dilated)
    # Get the height and width of the image
    # height, width = img.shape[:2]
    ## TODO Make this more efficient
    #  # Fix horizontal and vertical lines
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
    # h_lines_dilated = cv2.morphologyEx(h_lines_dilated, cv2.MORPH_CLOSE, horizontal_kernel, iterations=9)
    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
    # v_lines_dilated = cv2.morphologyEx(v_lines_dilated, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
    # if debug_level == 1:
    #     plshow("h_lines_dilated", h_lines_dilated)
    #     plshow("v_lines_dilated", v_lines_dilated)
    # Combine the dilated horizontal and vertical lines binary images
    lines_combined = cv2.bitwise_or(h_lines_dilated, v_lines_dilated)
    if debug_level == 1:
        plshow("lines_combines", lines_combined)
    # Erode the binary mask to remove noise and small details
    kernel_size = 3
    lines_eroded = cv2.erode(
        lines_combined,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
        iterations=7,
    )
    if debug_level == 1:
        plshow("lines_eroded", lines_eroded)
    # Apply the eroded binary mask to the original image
    if debug_level == 1:
        result = cv2.bitwise_and(img, img, mask=lines_eroded)
        plshow("result", result)
    cnts = cv2.findContours(lines_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Convert the tuple to a list
    cnts = list(cnts)
    # Remove the largest area contour
    largest_contour = max(cnts, key=cv2.contourArea)
    cnts.remove(largest_contour)
    cnts, _ = cntx.sort_contours(cnts, method="top-to-bottom")
    # row = []
    # rows = []
    # r=[]
    # for idx, cnt in enumerate(cnts):
    #     row.append(cnt)
    #     if (idx+1) % 9 == 0:
    #         cnts, _ = cntx.sort_contours(cnts, method="left-to-right")
    #         for cnt in cnts:
    #             r.append(cnt)

    #         row = []
    # if len(rows) == 3:
    #     print("True")

    shapes = []
    contours = []
    # So at this point, there should be 27 contours...double check the logic removing the largest contour
    # So we initialize each as a shape which gets the vertices based on the approximation of the contour.
    # We then sort the vertices so the top left is in the top left
    # We then generate a label position to be the top left 
    # once we have the vertices, we sort position based on them
    # the label position is
    for idx, cnt in enumerate(cnts):
        try:
            # cv2.drawContours(src_image, [cnt], -1, (0, 255, 0), 5)
            shape = Shape(cnt)
            shape.image = src_image
            
            
            shapes.append(shape)
            contours.append(cnt)

        except:
            cv2.drawContours(src_image, [cnt], -1, (0, 255, 0), 5)
            plshow("lines_combined", lines_combined)
            plshow("lines_eroded", lines_eroded)
            plshow("exception", src_image)
            print("errorrring")
            ## TODO Log Errors
            continue
    if debug_level == 1 or debug_level == 2:
        for idx, cnt in enumerate(cnts):
            shape.draw(
                            src_image,
                            label_shape=True,
                            label=str(idx),
                            draw_label_config=DrawConfigs.DEFAULT_LABEL,
                            draw_line_config=DrawConfigs.DEFAULT_LINE,
                            display_image=False,
                        )
        plshow("Final", src_image)

    return shapes, contours


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=None)

###code to do connected component visualisation
def imshow_components(labels):
    ### creating a hsv image, with a unique hue value for each label
    label_hue = np.uint8(179 * labels / np.max(labels))
    ### making saturation and volume to be 255
    empty_channel = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, empty_channel, empty_channel])
    ### converting the hsv image to BGR image
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    ### returning the color image for visualising Connected Componenets
    return labeled_img
