import cv2
import pandas as pd
import numpy as np
import math 
import imutils

def line_angle(line):
    return math.atan2(line[1] - line[3], line[0] - line[2])*180/math.pi

def switchHigherLowerVertical(data):
    if data['y1'] >= data['y2']:
        return data['y1'], data['y2']
    else:
        return data['y2'], data['y1']

def switchHigherLowerHorizontal(data):
    if data['x1'] >= data['x2']:
        return data['x2'], data['x2']
    else:
        return data['x1'], data['x2']

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def check_B_inside_A(A, B):
    # If rectangle B is inside rectangle A
    # bottomA <= topB
    if((A[0] <= B[0]) and (A[0]+A[2] >= B[0]+B[2]) and (A[1] <= B[1]) and (A[1]+A[3] >= B[1]+B[3])):
        return True
    else:
        return False

def line_detection(img_ori, img_gray, iterations=3):
    """
    Summary line.
    using series of cv2 methods, detect if there's combination of lines that possible create a image then decide wether that image is a table or not by counting the inner rectangle.

    Parameters:
    img_ori : return from cv2.imread
    img_gray : return from cv2.imread from grayscale
    iterations (int) : iteration to do erode and dilatation  

    Return:
    tables : list of list of integer, contain boundingboxes of table detected with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    non_table : list of list of integer, contain boundingboxes of non_table detected with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    """ 
    filename = ''
    df_boxes_outer_all = pd.DataFrame()
    df_line_horizontals = pd.DataFrame(columns=['filename', 'x1', 'x2', 'y'])
    df_line_verticals = pd.DataFrame(columns=['filename', 'y1', 'y2', 'x'])
    image = img_ori
    img = image
    gray = img_gray
    height_,width_ = gray.shape
    #thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # inverting the image
    img_bin = 255-img_bin
    kernel_len = np.array(img).shape[1]//100
    # Defining a horizontal kernel to detect all horizontal lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))  # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=iterations)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=iterations)
    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=iterations)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=iterations)
    # Eroding and thesholding the vertical lines
    img_v = cv2.erode(~vertical_lines, kernel, iterations=iterations)
    thresh, img_v = cv2.threshold(img_v, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    img_v = cv2.erode(img_v, kernel, iterations=iterations)
    # Eroding and thesholding the horizontal lines
    img_h = cv2.erode(~horizontal_lines, kernel, iterations=2)
    thresh, img_h = cv2.threshold(img_h, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = img_h
    # All Lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    lines = cv2.HoughLinesP(image=edges, rho=0.02, theta=np.pi/500, threshold=10,lines=np.array([]), minLineLength=minLineLength, maxLineGap=100)
    if lines is None:
        lines_detected = False
    else:
        lines_detected = True
    horizontal_detected = False
    if(lines_detected):
        tolerance = 5
        # Horizontal Only
        horizontal_lines = [list(line[0]) for line in lines if (abs(line_angle(line[0])) > 180-tolerance) and (abs(line_angle(line[0])) < 180+tolerance)]
        horizontal_detected = len(horizontal_lines) > 0
        if(horizontal_detected):
            df_horizontal = pd.DataFrame(horizontal_lines, columns=['x1', 'y1', 'x2', 'y2'])
            x1x2 = [list(x) for x in df_horizontal.apply(switchHigherLowerHorizontal, axis=1)]
            df_horizontal[['x1', 'x2']] = x1x2
            df_horizontal.sort_values(['y1', 'x1'], inplace=True)
            df_horizontal.reset_index(drop=True, inplace=True)
            y_th = 20
            separate_line_index = df_horizontal[df_horizontal.diff()['y1'] > y_th].index.tolist()
            separate_line_index = [0]+separate_line_index+[df_horizontal.shape[0]-1]
            line_index = []
            for i in range(len(separate_line_index)-1):
                for j in range(separate_line_index[i], separate_line_index[i+1]):
                    line_index.append(i)
            line_index_df = pd.DataFrame(line_index, columns=['line_index'])
            df_h = pd.concat([line_index_df, df_horizontal], axis=1)
            df_h.fillna(method='ffill', inplace=True)
            df_h_sort = pd.DataFrame(columns=df_h.columns)
            indexes = df_h['line_index'].unique()
            for index in indexes:
                df_temp = df_h[df_h['line_index'] == index].sort_values('x1')
                df_h_sort = pd.concat([df_h_sort, df_temp], axis=0)
            df_h = df_h_sort
            df_h.reset_index(drop=True, inplace=True)
            h_lines = list(df_h['line_index'].unique())
            line_no = 1
            df_line_no = pd.DataFrame(columns=['line_no'])
            for h_line in h_lines:
                line_no_list = []
                df_line_no_temp = pd.DataFrame(columns=['line_no'])
                df_temp = df_h[df_h['line_index'] == h_line]
                df_temp_x_sort = df_temp.sort_values(
                    'x1').reset_index(drop=True)
                max_x = df_temp_x_sort['x2'][0]
                min_column_width = 200
                for i in range(df_temp_x_sort.shape[0]):
                    if(df_temp_x_sort['x1'][i] <= max_x+min_column_width):
                        line_no_list.append(line_no)
                        if(max_x < df_temp_x_sort['x2'][i]):
                            max_x = df_temp_x_sort['x2'][i]
                    else:
                        line_no += 1
                        line_no_list.append(line_no)
                        max_x = df_temp_x_sort['x2'][i]
                df_line_no_temp['line_no'] = line_no_list
                df_line_no = pd.concat([df_line_no, df_line_no_temp], axis=0)
                line_no += 1
            df_line_no.reset_index(drop=True, inplace=True)
            df_h_final = pd.concat([df_h, df_line_no], axis=1)
            line_no = list(df_h_final['line_no'].unique())
            img_temp = img
            df_line_horizontal = pd.DataFrame(columns=['filename', 'x1', 'x2', 'y'])
            for line in line_no:
                x1 = df_h_final[df_h_final['line_no'] == line]['x1'].min()
                x2 = df_h_final[df_h_final['line_no'] == line]['x2'].max()
                y = int(df_h_final[df_h_final['line_no'] == line]['y1'].mean())
                cv2.line(img_temp, (x1, y), (x2, y),(0, 0, 255), 3, cv2.LINE_AA)
                df_line_horizontal.loc[df_line_horizontal.shape[0]] = [filename, x1, x2, y]
            df_line_horizontals = pd.concat([df_line_horizontals, df_line_horizontal], axis=0)
            df_line_horizontals.reset_index(inplace=True, drop=True)

    img = image
    gray = img_v
    # All Lines
    edges = cv2.Canny(gray, 225, 250, apertureSize=3)
    minLineLength = 50
    lines = cv2.HoughLinesP(image=edges, rho=0.02, theta=np.pi/500, threshold=10, lines=np.array([]), minLineLength=minLineLength, maxLineGap=100)
    # Vertical Only
    tolerance = 5
    vertical_detected = False
    if lines is None:
        lines_detected = False
    else:
        lines_detected = True
    if(lines_detected):
        vertical_lines = [list(line[0]) for line in lines if (abs(line_angle(line[0])) > 90-tolerance) and (abs(line_angle(line[0])) < 90+tolerance)]
        vertical_detected = len(vertical_lines) > 0
        if(vertical_detected):
            vertical_detected = len(lines) > 0
            df_vertical = pd.DataFrame(vertical_lines, columns=['x1', 'y1', 'x2', 'y2'])
            y1y2 = [list(x) for x in df_vertical.apply(switchHigherLowerVertical, axis=1)]
            df_vertical[['y1', 'y2']] = y1y2
            df_vertical.sort_values(['x1', 'y2'], inplace=True)
            df_vertical.reset_index(drop=True, inplace=True)
            x_th = 20
            separate_line_index = df_vertical[df_vertical.diff()['x1'] > x_th].index.tolist()
            separate_line_index = [0] + \
                separate_line_index+[df_vertical.shape[0]-1]
            line_index = []
            for i in range(len(separate_line_index)-1):
                for j in range(separate_line_index[i], separate_line_index[i+1]):
                    line_index.append(i)
            line_index_df = pd.DataFrame(line_index, columns=['line_index'])
            df_v = pd.concat([line_index_df, df_vertical], axis=1)
            df_v.fillna(method='ffill', inplace=True)
            df_v_sort = pd.DataFrame(columns=df_v.columns)
            indexes = df_v['line_index'].unique()
            for index in indexes:
                df_temp = df_v[df_v['line_index'] == index].sort_values('y2')
                df_v_sort = pd.concat([df_v_sort, df_temp], axis=0)
            df_v = df_v_sort
            df_v.reset_index(drop=True, inplace=True)
            v_lines = list(df_v['line_index'].unique())
            line_no = 1
            df_line_no = pd.DataFrame(columns=['line_no'])
            for v_line in v_lines:
                line_no_list = []
                df_line_no_temp = pd.DataFrame(columns=['line_no'])
                df_temp = df_v[df_v['line_index'] == v_line]
                df_temp_y_sort = df_temp.sort_values('y2').reset_index(drop=True)
                max_y = df_temp_y_sort['y1'][0]
                min_row_width = 100
                for i in range(df_temp_y_sort.shape[0]):
                    if(df_temp_y_sort['y2'][i] <= max_y+min_row_width):
                        line_no_list.append(line_no)
                        if(max_y < df_temp_y_sort['y1'][i]):
                            max_y = df_temp_y_sort['y1'][i]
                    else:
                        line_no += 1
                        line_no_list.append(line_no)
                        max_y = df_temp_y_sort['y1'][i]
                df_line_no_temp['line_no'] = line_no_list
                df_line_no = pd.concat([df_line_no, df_line_no_temp], axis=0)
                line_no += 1
            df_line_no.reset_index(drop=True, inplace=True)
            df_v_final = pd.concat([df_v, df_line_no], axis=1)
            line_no = list(df_v_final['line_no'].unique())
            img_temp = img
            df_line_vertical = pd.DataFrame(
                columns=['filename', 'y1', 'y2', 'x'])
            for line in line_no:
                y1 = int(df_v_final[df_v_final['line_no'] == line]['y1'].max())
                y2 = int(df_v_final[df_v_final['line_no'] == line]['y2'].min())
                x = int(df_v_final[df_v_final['line_no'] == line]['x1'].mean())
                cv2.line(img_temp, (x, y1), (x, y2),
                         (0, 0, 255), 3, cv2.LINE_AA)
                df_line_vertical.loc[df_line_vertical.shape[0]] = [
                    filename, y1, y2, x]
            df_line_verticals = pd.concat([df_line_verticals, df_line_vertical], axis=0)
            df_line_verticals.reset_index(inplace=True, drop=True)

    img = image
    # Horizontal Line
    if(horizontal_detected):
        for i in range(df_line_horizontal.shape[0]):
            df_temp = df_line_horizontal.loc[i]
            x1, x2, y = df_temp[['x1', 'x2', 'y']].values
            cv2.line(img, (x1, y), (x2, y), (0, 0, 255), 3, cv2.LINE_AA)
    # Vertical Line
    if(vertical_detected):
        for i in range(df_line_vertical.shape[0]):
            df_temp = df_line_vertical.loc[i]
            y1, y2, x = df_temp[['y1', 'y2', 'x']].values
            cv2.line(img, (x, y1), (x, y2), (0, 0, 255), 3, cv2.LINE_AA)

    blank_image = np.zeros(shape=list(image.shape), dtype=np.uint8)
    blank_image.fill(255)
    df_line_horizontal = df_line_horizontals[df_line_horizontals['filename'] == filename]
    df_line_vertical = df_line_verticals[df_line_verticals['filename'] == filename]
    df_line_horizontal.reset_index(drop=True, inplace=True)
    df_line_vertical.reset_index(drop=True, inplace=True)
    for i in range(df_line_horizontal.shape[0]):
        df_temp = df_line_horizontal.loc[i]
        x1, x2, y = df_temp[['x1', 'x2', 'y']].values
        cv2.line(blank_image, (x1, y), (x2, y), (0, 0, 0), 3, cv2.LINE_AA)
    for i in range(df_line_vertical.shape[0]):
        df_temp = df_line_vertical.loc[i]
        y1, y2, x = df_temp[['y1', 'y2', 'x']].values
        cv2.line(blank_image, (x, y1), (x, y2), (0, 0, 0), 3, cv2.LINE_AA)
    
    # find the contours of rectangle from the line outline
    img_vh = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
    img = img_gray
    bitxor = cv2.bitwise_xor(img, img_vh)
    # Detect contours for following box detection
    contours = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    # Retrieve Cell Position
    # Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]  # Get mean of heights
    mean = np.mean(heights)
    BoundingBoxes = [[filename]+list(boundingBox) for boundingBox in list(boundingBoxes)]
    df_boxes = pd.DataFrame(BoundingBoxes, columns=['filename', 'x', 'y', 'w', 'h'])
    df_boxes_copy = df_boxes.copy()
    h_max = 0.95*img.shape[0]
    h_min = height_//50
    w_min = width_//50
    df_boxes_content = df_boxes[(df_boxes['h'] < h_max) & (df_boxes['h'] > height_//100) & (df_boxes['w'] > width_//100)]
    content_index = df_boxes_content.index
    # Table Detection
    df_boxes = df_boxes[(df_boxes['h'] < h_max) & (df_boxes['h'] > h_min) & (df_boxes['w'] > w_min)]
    boxes_index = df_boxes.index
    # Remove cell inside another cell
    skip_inside_box_index_from_zero = []
    skip_inside_box_index = []
    for i in range(df_boxes.shape[0]-1):
        if i not in skip_inside_box_index_from_zero:
            for j in range(i+1, df_boxes.shape[0]):
                A = df_boxes.values[i][1:]
                B = df_boxes.values[j][1:]
                if(check_B_inside_A(A, B)):
                    skip_inside_box_index_from_zero.append(j)
                    skip_inside_box_index.append(boxes_index[j])
                elif(check_B_inside_A(B, A)):
                    skip_inside_box_index_from_zero.append(i)
                    skip_inside_box_index.append(boxes_index[i])
    df_boxes_outer = df_boxes[~df_boxes.index.isin(skip_inside_box_index)]
    df_boxes_outer_all = pd.concat([df_boxes_outer_all, df_boxes_outer], axis=0)
    df_boxes_final = df_boxes_outer
    table = []
    non_table = []
    count_table = 0
    count_nontable = 0
    # count the inner rectangle of each outer box
    for i in range(df_boxes_outer.shape[0]):
        df_temp = df_boxes_outer.values[i]
        # save image
        x = df_temp[1]
        y = df_temp[2]
        w = df_temp[3]
        h = df_temp[4]
        ############### COUNT INNER RECT FOR EACH OUTER BOX ############
        start_index = df_boxes_outer.index[i]
        if(i == df_boxes_outer.shape[0]-1):
            end_index = content_index[-1]
        else:
            end_index = df_boxes_outer.index[i+1]
        scan_index = [content for content in content_index if content > start_index and content < end_index]
        rects_inside_number = 0
        for index in scan_index:
            A = df_boxes_outer.values[i][1:]
            B = df_boxes_content.loc[index].values[1:]
            if(check_B_inside_A(A, B)):
                rects_inside_number += 1
        threshold_table = 5  # if inner_rect>threshold_table -> table, vice versa
        if(rects_inside_number >= threshold_table):
            table.append([])
            table[count_table] = [int(x), int(y), int(w), int(h)]
            count_table += 1
        else:
            non_table.append([])
            non_table[count_nontable] = [int(x), int(y), int(w), int(h)]
            count_nontable += 1
    return(table, non_table)
