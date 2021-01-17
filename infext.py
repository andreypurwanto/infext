import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import pytesseract
from pytesseract import Output
import pkg_resources
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image as Img
from PIL import ImageTk
import math
import imutils
import timeit


def inside_box_all(boundb, innerb):
    result = []
    for i in range(len(boundb)):
        if (boundb[i][1] <= innerb[1] and boundb[i][0] <= innerb[0] and innerb[0] <= boundb[i][0]+boundb[i][2] and innerb[1] <= boundb[i][1]+boundb[i][3]):
            result.append(True)
        else:
            result.append(False)

    result2 = result[0]
    for i in range(len(result)):
        result2 = result2 or result[i]

    return(result2)

def return_BoundingBox_Pytesseract(img_ori):
    img = img_ori
    # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img3 = cv2.medianBlur(img, 5)
    d2 = pytesseract.image_to_data(img3, output_type=Output.DATAFRAME)
    if (len(img.shape) == 2):
        height_, width_ = img.shape
    elif (len(img.shape) == 3):
        height_, width_, _ = img.shape
    tables2 = []
    df_tables = d2[(d2['height'] < (height_)) & (d2['height'] > (height_//15)) &
                   (d2['width'] > (width_//15)) & (d2['width'] < width_) & (d2['conf'] != -1)]
    for _, row in df_tables.iterrows():
        tables2.append([int(row.left), int(row.top),
                        int(row.width), int(row.height)])
    return(tables2)

def return_text_non_table(img_ori, img_gray, height_, width_, table_, nontable_):
    def f(x):
        return pd.Series(dict(left=x['left'].min(), top=x['top'].min(), width=x['width'].max(), height=x['height'].sum(), lines_combine="%s" % ', '.join(x['text'])))

    def f2(x):
        return pd.Series(dict(left=x['left'].min(), top=x['top'].min(), width=x['width'].max(), height=x['height'].sum(), lines_combine="%s" % ', '.join(x['lines_combine'])))
    
    img = img_ori
    img2 = img_gray
    img3 = cv2.medianBlur(img, 5)
    # d = pytesseract.image_to_data(img3, output_type=Output.DATAFRAME, config='-l eng+gcr --psm 3')
    d2 = pytesseract.image_to_data(img3, output_type=Output.DATAFRAME)
    debug_dataframe = pd.DataFrame(data=None, columns=d2.columns)
    # if (len(img.shape) == 2):
    #     height_, width_ = img.shape
    # elif (len(img.shape) == 3):
    #     height_, width_, _ = img.shape

    tables = table_+nontable_

    tables2 = []
    df_tables = d2[(d2['height'] < (height_)) & (d2['height'] > (height_//15)) & (d2['width'] > (width_//15)) & (d2['width'] < width_) & (d2['conf'] != -1)]
    for index, row in df_tables.iterrows():
        tables2.append([row.left, row.top, row.width, row.height])

    df_text = d2[(d2['conf']) != -1].reset_index(drop=True)
    df_text["text"] = df_text["text"].astype(str)
    extracted_text = []
    extracted_text.append([])
    first_row = True
    count = 0
    for index, row in df_text.iterrows():
        if(len(tables) == 0):
            if first_row:
                extracted_text[count].append(row.text)
                debug_dataframe = debug_dataframe.append(
                    row, ignore_index=True)
                previous_x = row.top
                # previous_count = count
                first_row = False
            else:
                if(previous_x-50 < row.top < previous_x+50):
                    extracted_text[count].append(row.text)
                    previous_x = row.top
                    debug_dataframe = debug_dataframe.append(
                        row, ignore_index=True)
                else:
                    count = count+1
                    extracted_text.append([])
                    extracted_text[count].append(row.text)
                    previous_x = row.top
                    debug_dataframe = debug_dataframe.append(
                        row, ignore_index=True)
        else:
            if ((not (inside_box_all(tables, [row.left, row.top, row.width, row.height]))) and (len(row.text) > 1)):
                if first_row:
                    extracted_text[count].append(row.text)
                    debug_dataframe = debug_dataframe.append(row, ignore_index=True)
                    previous_x = row.top
                    # previous_count = count
                    first_row = False
                else:
                    if(previous_x-50 < row.top < previous_x+50):
                        extracted_text[count].append(row.text)
                        previous_x = row.top
                        debug_dataframe = debug_dataframe.append(row, ignore_index=True)
                    else:
                        count = count+1
                        extracted_text.append([])
                        extracted_text[count].append(row.text)
                        previous_x = row.top
                        debug_dataframe = debug_dataframe.append(row, ignore_index=True)

    tes = dict(list(debug_dataframe.groupby('block_num')))

    for key in tes:
        tes[key] = dict(list(tes[key].groupby('par_num')))

    for key in tes:
        for key2 in tes[key]:
            tes[key][key2] = dict(list(tes[key][key2].groupby('line_num')))

    df_sentences_bb = pd.DataFrame(
        columns=['text', 'left', 'top', 'width', 'height', 'lines'])

    # text_bb = []
    # sentences_all = []
    # count_sentence_all = 0
    first_row = True
    # mean_before = 0
    # row_before = 0
    # index_ = 0
    last_top = 0
    temp = ''
    # new_temp = True
    list_sentence = []
    count_lines = 0
    for key in tes:
        for key2 in tes[key]:
            for key3 in tes[key][key2]:
                temp = ''
                for index, row in tes[key][key2][key3].iterrows():
                    temp = temp + row.text + ' '
                    # row_before = row.top
                sum_number = 0
                for i in temp:
                    if i.isnumeric():
                        sum_number += 1
                if sum_number > 1 or len(temp) > 4:
                    if (tes[key][key2][key3]['top'].min() - last_top <= width_//36):
                        list_sentence.append({'text': temp, 'left': tes[key][key2][key3]['left'].min(), 'top': tes[key][key2][key3]['top'].min(), 'width': tes[key][key2][key3]['width'].sum(), 'height': tes[key][key2][key3]['height'].max(), 'lines': count_lines})
                        df_sentences_bb = df_sentences_bb.append({'text': temp, 'left': tes[key][key2][key3]['left'].min(), 'top': tes[key][key2][key3]['top'].min(), 'width': tes[key][key2][key3]['width'].sum(), 'height': tes[key][key2][key3]['height'].max(), 'lines': count_lines}, ignore_index=True)
                    else:
                        count_lines += 1
                        list_sentence.append({'text': temp, 'left': tes[key][key2][key3]['left'].min(), 'top': tes[key][key2][key3]['top'].min(), 'width': tes[key][key2][key3]['width'].sum(), 'height': tes[key][key2][key3]['height'].max(), 'lines': count_lines})
                        df_sentences_bb = df_sentences_bb.append({'text': temp, 'left': tes[key][key2][key3]['left'].min(), 'top': tes[key][key2][key3]['top'].min(), 'width': tes[key][key2][key3]['width'].sum(), 'height': tes[key][key2][key3]['height'].max(), 'lines': count_lines}, ignore_index=True)

                    last_top = tes[key][key2][key3]['top'].min()
                    last_temp = temp

    df_combined_sentences_bb = df_sentences_bb.groupby('lines').apply(f)
    df_combined_sentences_bb = df_combined_sentences_bb.reset_index()
    df_combined_sentences_bb['lines_par'] = 0
    lines_par_count = 1
    bbox_text = []
    blur = cv2.GaussianBlur(img2, (7, 7), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=20)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        check = True
        x, y, w, h = cv2.boundingRect(c)
        # print((height_*width_//10000))
        if(w*h > (height_*width_//500)):
            for c2 in cnts:
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                if (w2*h2 > (height_*width_//500)) and (w2*h2 > w*h) and (inside_box_all([[x2, y2, w2, h2]], [x, y, w, h]) or get_iou([x2, y2, w2, h2], [x, y, w, h]) > 0 or get_iou([x, y, w, h], [x2, y2, w2, h2]) > 0):
                    check = False
                    break
            if check:
                bbox_text.append([x, y, w, h])
                check = True
                for index, row in df_combined_sentences_bb.iterrows():

                    if row.lines_par == 0:
                        if get_iou([row['left'], row['top'], row['width'], row['height']], [x, y, w, h]) > 0:
                            if check:
                                df_combined_sentences_bb.loc[index,'lines_par'] = lines_par_count
                                lines_par_count += 1
                                check = False
                            else:
                                df_combined_sentences_bb.loc[index,'lines_par'] = lines_par_count-1

    list_df_text_grouped = []
    for key, df_lines_par in df_combined_sentences_bb.groupby('lines_par'):

        df_lines_par = df_lines_par.reset_index(drop=True)
        df_lines_par['lowercase_tag'] = 0
        count_lowercase_tag = 0

        for index, row in (df_lines_par).iterrows():
            if (index > 0):
                if row.lines_combine[0].islower():
                    df_lines_par.loc[index,'lowercase_tag'] = count_lowercase_tag
                else:
                    count_lowercase_tag += 1
                    df_lines_par.loc[index,'lowercase_tag'] = count_lowercase_tag

        df_lines_par = df_lines_par.groupby('lowercase_tag').apply(f2)
        df_lines_par = df_lines_par.reset_index()
        if (len(df_lines_par) == 1):
            df_lines_par['detected_as'] = 'text'
        else:
            df_lines_par['detected_as'] = 'text'
            df_lines_par.loc[0, 'detected_as'] = 'title'
        df_lines_par = df_lines_par.drop(columns='lowercase_tag')
        list_df_text_grouped.append(df_lines_par)

    list_df_text_grouped.reverse()
    for i in range(len(list_df_text_grouped)):
        list_df_text_grouped[i]['group'] = i+1
    if len(list_df_text_grouped)>0:
        df_combined_final = pd.concat(list_df_text_grouped)
    else:
        df_combined_final = pd.DataFrame(columns = ['left','top','width','height','lines_combine','detected_as','group'])
    # df_combined_final = pd.concat(list_df_text_grouped)
    return(df_combined_sentences_bb, list_df_text_grouped, df_combined_final)

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

def check_B_inside_A(A, B):
    # If rectangle B is inside rectangle A
    # bottomA <= topB
    if((A[0] <= B[0]) and (A[0]+A[2] >= B[0]+B[2]) and (A[1] <= B[1]) and (A[1]+A[3] >= B[1]+B[3])):
        return True
    else:
        return False

def get_lines(img_ori, img_gray):
    filename = ''
    df_boxes_outer_all = pd.DataFrame()
    df_info_all = pd.DataFrame()
    df_line_horizontals = pd.DataFrame(columns=['filename', 'x1', 'x2', 'y'])
    df_line_verticals = pd.DataFrame(columns=['filename', 'y1', 'y2', 'x'])
    # image_2 = cv2.imread(file)
    image = img_ori
    img = image
    gray = img_gray

    # img.shape#thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # inverting the image
    img_bin = 255-img_bin

    # Length(width) of kernel as 100th of total width
    # Defining a vertical kernel to detect all vertical lines of image
    kernel_len = np.array(img).shape[1]//100
    # Defining a horizontal kernel to detect all horizontal lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_len, 1))  # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

    # Eroding and thesholding the vertical lines
    img_v = cv2.erode(~vertical_lines, kernel, iterations=2)
    thresh, img_v = cv2.threshold(
        img_v, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    img_v = cv2.erode(img_v, kernel, iterations=1)
    # file_dir = os.path.join(only_vertical_folder,filename)
    # cv2.imwrite(file_dir, img_v)

    # Eroding and thesholding the horizontal lines
    img_h = cv2.erode(~horizontal_lines, kernel, iterations=2)
    thresh, img_h = cv2.threshold(
        img_h, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # file_dir = os.path.join(only_horizontal_folder,filename)
    # cv2.imwrite(file_dir, img_h)

    ########################################### Horizontal Detection ##############################################
    gray = img_h
    # All Lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    lines = cv2.HoughLinesP(image=edges, rho=0.02, theta=np.pi/500, threshold=10,
                            lines=np.array([]), minLineLength=minLineLength, maxLineGap=100)

    if lines is None:
        lines_detected = False
    else:
        lines_detected = True

    horizontal_detected = False
    if(lines_detected):
        tolerance = 5
        # Horizontal Only
        horizontal_lines = [list(line[0]) for line in lines if (abs(line_angle(
            line[0])) > 180-tolerance) and (abs(line_angle(line[0])) < 180+tolerance)]
        horizontal_detected = len(horizontal_lines) > 0
        if(horizontal_detected):
            df_horizontal = pd.DataFrame(horizontal_lines, columns=[
                                         'x1', 'y1', 'x2', 'y2'])

            x1x2 = [list(x) for x in df_horizontal.apply(
                switchHigherLowerHorizontal, axis=1)]
            df_horizontal[['x1', 'x2']] = x1x2
            df_horizontal.sort_values(['y1', 'x1'], inplace=True)
            df_horizontal.reset_index(drop=True, inplace=True)

            y_th = 20
            separate_line_index = df_horizontal[df_horizontal.diff()[
                'y1'] > y_th].index.tolist()
            separate_line_index = [
                0]+separate_line_index+[df_horizontal.shape[0]-1]
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
                        #                 print(i)
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

            df_line_horizontal = pd.DataFrame(
                columns=['filename', 'x1', 'x2', 'y'])
            for line in line_no:
                x1 = df_h_final[df_h_final['line_no'] == line]['x1'].min()
                x2 = df_h_final[df_h_final['line_no'] == line]['x2'].max()
                y = int(df_h_final[df_h_final['line_no'] == line]['y1'].mean())
                cv2.line(img_temp, (x1, y), (x2, y),
                         (0, 0, 255), 3, cv2.LINE_AA)
                df_line_horizontal.loc[df_line_horizontal.shape[0]] = [
                    filename, x1, x2, y]

            df_line_horizontals = pd.concat(
                [df_line_horizontals, df_line_horizontal], axis=0)
            df_line_horizontals.reset_index(inplace=True, drop=True)

    ########################################### Vertical Detection ##############################################
    # image = cv2.imread(file)
    img = image

    gray = img_v
    # All Lines
    edges = cv2.Canny(gray, 225, 250, apertureSize=3)
    minLineLength = 50

    lines = cv2.HoughLinesP(image=edges, rho=0.02, theta=np.pi/500, threshold=10,
                            lines=np.array([]), minLineLength=minLineLength, maxLineGap=100)
    # Vertical Only
    tolerance = 5
    vertical_detected = False

    if lines is None:
        lines_detected = False
    else:
        lines_detected = True
    if(lines_detected):
        vertical_lines = [list(line[0]) for line in lines if (abs(line_angle(
            line[0])) > 90-tolerance) and (abs(line_angle(line[0])) < 90+tolerance)]
        vertical_detected = len(vertical_lines) > 0
        if(vertical_detected):
            vertical_detected = len(lines) > 0
            df_vertical = pd.DataFrame(vertical_lines, columns=[
                                       'x1', 'y1', 'x2', 'y2'])

            y1y2 = [list(x) for x in df_vertical.apply(
                switchHigherLowerVertical, axis=1)]
            df_vertical[['y1', 'y2']] = y1y2

            df_vertical.sort_values(['x1', 'y2'], inplace=True)
            df_vertical.reset_index(drop=True, inplace=True)
            x_th = 20
            separate_line_index = df_vertical[df_vertical.diff()[
                'x1'] > x_th].index.tolist()
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
                df_temp_y_sort = df_temp.sort_values(
                    'y2').reset_index(drop=True)
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

    ########################################### Combine Detection ##############################################
    # image = cv2.imread(file)
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
    # bitnot = cv2.bitwise_not(bitxor)  # Plotting the generated image
    # Detect contours for following box detection
    contours = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    # Retrieve Cell Position
    # Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3]
               for i in range(len(boundingBoxes))]  # Get mean of heights
    mean = np.mean(heights)
    BoundingBoxes = [[filename]+list(boundingBox)
                     for boundingBox in list(boundingBoxes)]
    df_boxes = pd.DataFrame(BoundingBoxes, columns=[
                            'filename', 'x', 'y', 'w', 'h'])

    h_max = 0.95*img.shape[0]

    h_min = 250
    w_min = 200
    df_boxes_content = df_boxes[(df_boxes['h'] < h_max) & (
        df_boxes['h'] > 50) & (df_boxes['w'] > 100)]
    content_index = df_boxes_content.index

    # Table Detection
    df_boxes = df_boxes[(df_boxes['h'] < h_max) & (
        df_boxes['h'] > h_min) & (df_boxes['w'] > w_min)]

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

    df_boxes_outer_all = pd.concat(
        [df_boxes_outer_all, df_boxes_outer], axis=0)
    df_boxes_final = df_boxes_outer
    # FinalBoundingBoxes = df_boxes_final.values

    # ######################### Save all outer box #########################################
    # box = []# Get position (x,y), width and height for every contour and show the contour on image
    # i=0
    # for _,x,y,w,h in FinalBoundingBoxes:
    #     filename_out = filename.split('.')[:-1][0]+'_table_'+str(i)+'.png'
    #     out_file_table_name = os.path.join(out_all_folder,filename_out)
    #     image = cv2.imread(file)
    #     img = image
    #     image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
    #     box.append([x,y,w,h])
    #     cv2.imwrite(out_file_table_name,img)
    #     i+=1

    ########################## Classify table and non-table by number of inner rectangle #########
    table = []
    non_table = []
    count_table = 0
    count_nontable = 0
    # divider = 35

    # count the inner rectangle of each outer box
    for i in range(df_boxes_outer.shape[0]):

        df_info = pd.DataFrame(
            columns=["filename", "wh_ratio", "area", "n_rect", ])
        df_temp = df_boxes_outer.values[i]

        # save image
        x = df_temp[1]
        y = df_temp[2]
        w = df_temp[3]
        h = df_temp[4]

        ############### COUNT INNER RECT FOR EACH OUTER BOX ############
        # data = [filename, w/h ratio, w*h area]
        data_info = [filename, w/h, w*h]
        start_index = df_boxes_outer.index[i]
        if(i == df_boxes_outer.shape[0]-1):
            end_index = content_index[-1]
        else:
            end_index = df_boxes_outer.index[i+1]
        scan_index = [content for content in content_index if content >
                      start_index and content < end_index]
        rects_inside_number = 0
        for index in scan_index:
            A = df_boxes_outer.values[i][1:]
            B = df_boxes_content.loc[index].values[1:]
            if(check_B_inside_A(A, B)):
                rects_inside_number += 1
        # append number of rect
        data_info.append(rects_inside_number)
        df_info.loc[start_index] = data_info

        df_info_all = pd.concat([df_info_all, df_info], axis=0)
        # category = ''
        image = img_ori
        # print(image.shape)
        if (len(image.shape) == 2):
            (height_, width_) = img.shape
        elif (len(image.shape) == 3):
            (height_, width_) = img.shape
        ################ SAVE TABLE DETECTION ######################

        # filename_out = filename.split('.')[:-1][0]+'_table_'+str(i)+'.png'
        threshold_table = 5  # if inner_rect>threshold_table -> table, vice versa
        if(rects_inside_number >= threshold_table):
            # out_file_table_name = os.path.join(out_table_folder,filename_out)
            # category = 'table'
            # print('-----------------TABLE--------------------')
            # # cv2_imshow(image[(y-(height_//divider)):(y-(height_//divider))+(h+(height_//divider)*4//2), (x-(width_//divider)):(x-(width_//divider))+(w+(width_//divider)*3)])
            # print('-----------------TABLE--------------------')
            table.append([])
            table[count_table] = [int(x), int(y), int(w), int(h)]
            count_table += 1
        else:
            # out_file_table_name = os.path.join(out_non_table_folder,filename_out)
            # category = 'non_table'
            # print('-----------------NONTABLE--------------------')
            # # cv2_imshow(image[(y-(height_//divider)):(y-(height_//divider))+(h+(height_//divider)*4//2), (x-(width_//divider)):(x-(width_//divider))+(w+(width_//divider)*3)])
            # print('-----------------NONTABLE--------------------')
            non_table.append([])
            non_table[count_nontable] = [int(x), int(y), int(w), int(h)]
            count_nontable += 1

        # img = image
        # image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # box.append([x,y,w,h])
        # cv2.imwrite(out_file_table_name,img)
        # index_box +=1

        # df_table_classification.loc[df_table_classification.shape[0]]= [filename, i,rects_inside_number, category]
        # table_classification = 'table_classification.csv'
        # table_classification_file = os.path.join(lines_boxes_dir, table_classification)
        # df_table_classification.to_csv(table_classification_file)

    return(table, non_table)

def create_table(tables, img_gray, size_):
    # img_ori = img_ori
    img = img_gray
    # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img3 = cv2.medianBlur(img, 5)
    # d = pytesseract.image_to_data(img3, output_type=Output.DATAFRAME, config ='--psm 3')
    # d['area'] = d['width']*d['height']
    dataframe = []
    tables_img = []
    # tables_img_ori = []
    for i in range(len(tables)):
        # print(i)
        tables_img.append(img[tables[i][1]:tables[i][1]+tables[i][3], tables[i][0]:tables[i][0]+tables[i][2]])
        # tables_img_ori.append(img[tables[i][1]:tables[i][1]+tables[i][3], tables[i][0]:tables[i][0]+tables[i][2]])
    for x_image in range(len(tables_img)):
        try:
            img = tables_img[x_image]
            # print(type(img))
            # thresholding the image to a binary image
            thresh, img_bin = cv2.threshold(
                img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # img_bin = images[0]
            # img = images[0]
            # inverting the image
            img_bin = 255-img_bin

            # Plotting the image to see the output
            plotting = plt.imshow(img_bin, cmap='gray')
            # plt.show()

            # countcol(width) of kernel as 100th of total width
            kernel_len = np.array(img).shape[1]//25
            # print('kernel_len',kernel_len)
            # Defining a vertical kernel to detect all vertical lines of image

            ver_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, kernel_len))
            # Defining a horizontal kernel to detect all horizontal lines of image
            hor_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (kernel_len, 1))
            # A kernel of 2x2
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

            # Use vertical kernel to detect and save the vertical lines in a jpg
            image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
            vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

            # print(vertical_lines)
            # cv2.imwrite("vertical.jpg",vertical_lines)
            # Plot the generated image
            plotting = plt.imshow(image_1, cmap='gray')
            # plt.show()

            # Use horizontal kernel to detect and save the horizontal lines in a jpg
            image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
            horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
            # cv2.imwrite("horizontal.jpg",horizontal_lines)
            # Plot the generated image
            plotting = plt.imshow(image_2, cmap='gray')
            # plt.show()

            # Combine horizontal and vertical lines in a new third image, with both having same weight.
            img_vh = cv2.addWeighted(
                vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
            # Eroding and thresholding the image
            img_vh = cv2.erode(~img_vh, kernel, iterations=2)
            thresh, img_vh = cv2.threshold(
                img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # cv2.imwrite("img_vh.jpg", img_vh)
            bitxor = cv2.bitwise_xor(img, img_vh)
            bitnot = cv2.bitwise_not(bitxor)
            # Plotting the generated image

            # Detect contours for following box detection
            contours, hierarchy = cv2.findContours(
                img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print('contours',contours)
            # print('hierarchy',hierarchy)
            # Sort all the contours by top to bottom.
            contours, boundingBoxes = sort_contours(
                contours, method="top-to-bottom")
            # print('contours',contours)
            # print('hierarchy',hierarchy)
            # Creating a list of heights for all detected boxes
            heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

            # Get mean of heights
            mean = np.mean(heights)
            # Create list box to store all boxes in
            box = []
            # Get position (x,y), width and height for every contour and show the contour on image
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if ((w < size_/2 and h < size_/3)):
                    image = cv2.rectangle(
                        img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    box.append([x, y, w, h])

            # plotting = plt.imshow(image,cmap='gray')
            # plt.show()

            # Creating two lists to define row and column in which cell is located
            row = []
            column = []
            j = 0

            # Sorting the boxes to their respective row and column
            for i in range(len(box)):

                if(i == 0):
                    column.append(box[i])
                    previous = box[i]

                else:
                    if(box[i][1] <= previous[1]+mean/8):
                        column.append(box[i])
                        previous = box[i]

                        if(i == len(box)-1):
                            row.append(column)

                    else:
                        row.append(column)
                        column = []
                        previous = box[i]
                        column.append(box[i])

            # calculating maximum number of cells
            countcol = 0
            for i in range(len(row)):
                countcol = len(row[i])
                if countcol > countcol:
                    countcol = countcol
            # print('row = ',row)
            # print('column = ',column)
            # print('countcol = ',countcol)
            # Retrieving the center of each column
            center = [int(row[i][j][0]+row[i][j][2]/2)
                      for j in range(len(row[i])) if row[0]]

            center = np.array(center)
            center.sort()
            # print(center)
            # Regarding the distance to the columns center, the boxes are arranged in respective order

            finalboxes = []
            for i in range(len(row)):
                lis = []
                for k in range(countcol):
                    lis.append([])
                for j in range(len(row[i])):
                    diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
                    minimum = min(diff)
                    indexing = list(diff).index(minimum)
                    lis[indexing].append(row[i][j])
                finalboxes.append(lis)

            # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
            outer = []
            count_row = []
            count_row_ = 0
            check = 0
            for i in range(len(finalboxes)):
                for j in range(len(finalboxes[i])):
                    count_row_ = count_row_+1
                    inner = ''
                    if(len(finalboxes[i][j]) == 0):
                        outer.append(' ')
                    else:
                        for k in range(len(finalboxes[i][j])):
                            y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][
                                k][1], finalboxes[i][j][k][2], finalboxes[i][j][k][3]
                            finalimg = bitnot[x:x+h, y:y+w]
                    #   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                    #   border = cv2.copyMakeBorder(finalimg,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                    #   resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    #   dilation = cv2.dilate(resizing, kernel,iterations=2)
                    #   erosion = cv2.erode(dilation, kernel,iterations=2)
                            # out = pytesseract.image_to_string(erosion)
                            blur = cv2.GaussianBlur(finalimg, (7, 7), 0)
                            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                            dilate = cv2.dilate(thresh, kernel, iterations=20)
                            cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                            # cv2_imshow(dilate)
                            hmax = 0
                            wmax = 0
                            for c in cnts:
                                check = True
                                x, y, w, h = cv2.boundingRect(c)
                                # print(x,y,w,h)
                                if h > hmax:
                                  hmax = h
                                if w > wmax:
                                  wmax = w
                            if int(wmax*1.1)<hmax:
                                finalimg = cv2.rotate(finalimg, cv2.cv2.ROTATE_90_CLOCKWISE)
                            out = pytesseract.image_to_string(
                                finalimg, config='-l eng+gcr --psm 6')
                            if(len(out) < 2):
                                out = pytesseract.image_to_string(
                                    finalimg, config='-l eng+gcr --psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
                            out = out[:-1]
                            inner = inner + " " + out
                            # cv2_imshow(finalimg)
                        check = check+1
                        outer.append(inner.replace("\n", " "))
                        # print(inner.replace("\n"," "))
            count_row.append(count_row_)
            arr = np.array(outer)
            dataframe.append(pd.DataFrame(arr.reshape(len(row), countcol)))
        except IndexError:
            pass

    return(dataframe)

def save_img_table(tables, img_ori, name, page, dir_file, dir_next):
    # img_ori = cv2.imread(file)
    os.chdir(dir_next)
    # dataframe = []
    tables_img = []
    tables_img_ori = []
    for i in range(len(tables)):
        tables_img_ori.append(img_ori[tables[i][1]:tables[i][1]+tables[i][3], tables[i][0]:tables[i][0]+tables[i][2]])
        im = Img.fromarray(tables_img_ori[i])
        im.save(str(name)+"_page_"+str(page)+"_"+str(i)+".jpeg")
    os.chdir(dir_file)

def save_img_table2(tables, img_ori, name, page, dir_file, dir_next):
    # img_ori = cv2.imread(file)
    os.chdir(dir_next)
    # dataframe = []
    tables_img = []
    tables_img_ori = []
    for i in range(len(tables)):
        tables_img_ori.append(img_ori[tables[i][1]:tables[i][1]+tables[i][3], tables[i][0]:tables[i][0]+tables[i][2]])
        im = Img.fromarray(tables_img_ori[i])
        im.save(str(name)+".jpeg")
    os.chdir(dir_file)

def get_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])
    if ((x_right - x_left) < 0 and (y_bottom - y_top) < 0):
        intersection_area = (x_right - x_left) * (y_bottom - y_top) * (-1)
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # print(intersection_area)
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]
    # iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    iou = intersection_area / float(bb2_area)
    return iou

def combined_tables_and_nontable(table, nontable, cek_table, height_, width_):
    bbox_pytesseract = cek_table
    bbox_linedetection = table+nontable
    # img_ori = cv2.imread(file)
    # if (len(img_ori.shape) == 2):
    #     height_, width_ = img_ori.shape
    # elif (len(img_ori.shape) == 3):
    #     height_, width_, _ = img_ori.shape
    delete_bbox_linedetection = []
    delete_bbox_pytesseract = []
    for i in range(len(bbox_linedetection)):
        for j in range(len(bbox_linedetection)):
            if i != j:
                if get_iou(bbox_linedetection[i], bbox_linedetection[j]) > 0.8 and (bbox_linedetection[j] not in delete_bbox_linedetection):
                    delete_bbox_linedetection.append(bbox_linedetection[j])

    for i in range(len(bbox_pytesseract)):
        for j in range(len(bbox_pytesseract)):
            if i != j:
                if get_iou(bbox_pytesseract[i], bbox_pytesseract[j]) > 0.8 and (bbox_pytesseract[j] not in delete_bbox_pytesseract):
                    delete_bbox_pytesseract.append(bbox_pytesseract[j])

    for i in range(len(table)):
        for j in range(len(bbox_pytesseract)):
            if get_iou(table[i], bbox_pytesseract[j]) > 0.8 and (bbox_pytesseract[j] not in delete_bbox_pytesseract):
                delete_bbox_pytesseract.append(bbox_pytesseract[j])

    for i in range(len(table)):
        for j in range(len(bbox_pytesseract)):
            if get_iou(bbox_pytesseract[j], table[i]) > 0.8 and (bbox_pytesseract[j] not in delete_bbox_pytesseract):
                delete_bbox_pytesseract.append(bbox_pytesseract[j])

    for i in range(len(nontable)):
        for j in range(len(bbox_pytesseract)):
            if get_iou(nontable[i], bbox_pytesseract[j]) > 0.8:
                if (nontable[i][2]*nontable[i][3] < bbox_pytesseract[j][2]*bbox_pytesseract[j][3]) and (nontable[i] not in delete_bbox_linedetection):
                    delete_bbox_linedetection.append(nontable[i])
                elif (nontable[i][2]*nontable[i][3] > bbox_pytesseract[j][2]*bbox_pytesseract[j][3]) and (bbox_pytesseract[j] not in delete_bbox_pytesseract):
                    delete_bbox_pytesseract.append(bbox_pytesseract[j])

    for i in range(len(nontable)):
        for j in range(len(bbox_pytesseract)):
            if get_iou(bbox_pytesseract[j], nontable[i]) > 0.8:
                if ((nontable[i][2]*nontable[i][3]) < (bbox_pytesseract[j][2]*bbox_pytesseract[j][3])) and (nontable[i] not in delete_bbox_linedetection):
                    delete_bbox_linedetection.append(nontable[i])
                elif (nontable[i][2]*nontable[i][3] > bbox_pytesseract[j][2]*bbox_pytesseract[j][3]) and (bbox_pytesseract[j] not in delete_bbox_pytesseract):
                    delete_bbox_pytesseract.append(bbox_pytesseract[j])

    for i in range(len(delete_bbox_linedetection)):
        bbox_linedetection.remove(delete_bbox_linedetection[i])

    for i in range(len(delete_bbox_pytesseract)):
        bbox_pytesseract.remove(delete_bbox_pytesseract[i])

    combine_x = []
    count_combine_x = 0
    bbox_total = bbox_pytesseract + bbox_linedetection
    first = True
    for i in range(len(bbox_total)):
        for j in range(len(bbox_total)):
            for k in range(2):
                if k == 0:
                    distance_x = bbox_total[i][0] - \
                        (bbox_total[j][0] + bbox_total[j][2])
                else:
                    distance_x = (
                        bbox_total[i][0] + bbox_total[i][2]) - bbox_total[j][0]
                if (abs(distance_x) < (width_//50)) and (bbox_total[j][1] <= bbox_total[i][1] <= bbox_total[j][1]+bbox_total[j][3]) and (bbox_total[j][1] <= bbox_total[i][1]+bbox_total[i][3] <= bbox_total[j][1]+bbox_total[j][3]):
                    if([j, i] not in combine_x):
                        if first:
                            combine_x.append([])
                            combine_x[count_combine_x] = [j, i]
                            count_combine_x += 1
                            first = False
                        else:
                            check = True
                            for l in range(len(combine_x)):
                                if j in combine_x[l]:
                                    if i not in combine_x[l]:
                                        combine_x[l].append(i)
                                    check = False
                                    break
                                if i in combine_x[l]:
                                    if j not in combine_x[l]:
                                        combine_x[l].append(j)
                                    check = False
                                    break
                            if check:
                                combine_x.append([])
                                combine_x[count_combine_x] = [j, i]
                                count_combine_x += 1

    bbox_combine_x = []
    for i in range(len(combine_x)):
        tes = {'x': [], 'y': [], 'w': [], 'h': []}
        for j in range(len(combine_x[i])):
            tes['x'].append(bbox_total[combine_x[i][j]][0])
            tes['y'].append(bbox_total[combine_x[i][j]][1])
            tes['w'].append(bbox_total[combine_x[i][j]][2])
            tes['h'].append(bbox_total[combine_x[i][j]][3])
        bbox_combine_x.append([min(tes['x']), min(tes['y']), sum(tes['w']), max(tes['h'])])

    delete_final = []
    for i in range(len(combine_x)):
        for j in range(len(combine_x[i])):
            if bbox_total[combine_x[i][j]] not in delete_final:
                delete_final.append(bbox_total[combine_x[i][j]])

    for i in range(len(delete_final)):
        if delete_final[i] in bbox_total:
            bbox_total.remove(delete_final[i])

    for i in range(len(bbox_combine_x)):
        if bbox_combine_x[i] not in bbox_total:
            bbox_total.append(bbox_combine_x[i])

    combine_y = []
    count_combine_y = 0
    first = True
    for i in range(len(bbox_total)):
        for j in range(len(bbox_total)):
            for k in range(2):
                if k == 0:
                    distance_y = bbox_total[i][1] - \
                        (bbox_total[j][1] + bbox_total[j][3])
                else:
                    distance_y = (bbox_total[i][1] + bbox_total[i][3]) - bbox_total[j][1]
                if (abs(distance_y) <= (height_//50)) and (bbox_total[j][0] <= bbox_total[i][0] <= bbox_total[j][0]+bbox_total[j][2]) and (bbox_total[j][0] <= bbox_total[i][0]+bbox_total[i][2] <= bbox_total[j][0]+bbox_total[j][2]):
                    if([j, i] not in combine_y):
                        if first:
                            combine_y.append([])
                            combine_y[count_combine_y] = [j, i]
                            count_combine_y += 1
                            first = False
                        else:
                            check = True
                            for l in range(len(combine_y)):
                                if j in combine_y[l]:
                                    if i not in combine_y[l]:
                                        combine_y[l].append(i)
                                    check = False
                                    break
                                if i in combine_y[l]:
                                    if j not in combine_y[l]:
                                        combine_y[l].append(j)
                                    check = False
                                    break
                            if check:
                                combine_y.append([])
                                combine_y[count_combine_y] = [j, i]
                                count_combine_y += 1

    bbox_combine_y = []
    for i in range(len(combine_y)):
        tes = {'x': [], 'y': [], 'w': [], 'h': []}
        for j in range(len(combine_y[i])):
            tes['x'].append(bbox_total[combine_y[i][j]][0])
            tes['y'].append(bbox_total[combine_y[i][j]][1])
            tes['w'].append(bbox_total[combine_y[i][j]][2])
            tes['h'].append(bbox_total[combine_y[i][j]][3])
        bbox_combine_y.append(
            [min(tes['x']), min(tes['y']), max(tes['w']), sum(tes['h'])])

    delete_final = []
    for i in range(len(combine_y)):
        for j in range(len(combine_y[i])):
            if bbox_total[combine_y[i][j]] not in delete_final:
                delete_final.append(bbox_total[combine_y[i][j]])

    for i in range(len(delete_final)):
        if delete_final[i] in bbox_total:
            bbox_total.remove(delete_final[i])

    for i in range(len(bbox_combine_y)):
        if bbox_combine_y[i] not in bbox_total:
            bbox_total.append(bbox_combine_y[i])
    check_tables = True
    dict_bbox_total_detected = {}
    for i in range(len(bbox_total)):
        for j in range(len(table)):
            if table[j] == bbox_total[i]:
                dict_bbox_total_detected[i] = []
                dict_bbox_total_detected[i].append(bbox_total[i])
                dict_bbox_total_detected[i].append('table')
                check_tables = False
                break
        if check_tables:
            dict_bbox_total_detected[i] = []
            dict_bbox_total_detected[i].append(bbox_total[i])
            dict_bbox_total_detected[i].append('non_table')
        check_tables = True
    # print(dict_bbox_total_detected)
    return (bbox_total, dict_bbox_total_detected)

def get_title_from_images(final_box, df_combined_sentences_bb, img_ori, height_, width_):
    # img_ori = cv2.imread(file)
    # final_box = final_box
    # print(final_box)
    # print(final_box.type())
    # final_box_copy = final_box.copy()
    df_image_title = pd.DataFrame(
        columns=['image_name', 'left', 'top', 'width', 'height', 'title', 'detected_as'])
    divider = 18
    temp = [0,0,0,0]
    count_table = 0
    count_nontable = 0
    for i in range(len(final_box)):
        get_title = False
        if (final_box[i][0][0] - (height_//divider)) < 0:
            # final_box[i][0][0] = 0
            temp[0] = 0
        else:
            # final_box[i][0][0] = final_box[i][0][0] - (height_//divider)
            temp[0] = final_box[i][0][0] - (height_//divider)
        if (final_box[i][0][1] - (width_//divider)) < 0:
            # final_box[i][0][1] = 0
            temp[1] = 0
        else:
            # final_box[i][0][1] = final_box[i][0][1] - (height_//divider)
            temp[1] = final_box[i][0][1] - (height_//divider)
        # final_box[i][0][2] = final_box[i][0][2] + (width_//divider)*1.8
        temp[2] = final_box[i][0][2] + (width_//divider)*1.8
        # final_box[i][0][3] = final_box[i][0][3] + (height_//divider)*1.8
        temp[3] = final_box[i][0][3] + (height_//divider)*1.8
        
        for index, row in df_combined_sentences_bb.iterrows():
            if inside_box_all([temp], [row.left, row.top, row.width, row.height]) and ('Fig. ' in row.lines_combine or 'Table ' in row.lines_combine):
                df_image_title = df_image_title.append({'image_name': 'combined_'+str(i), 'left': final_box[i][0][0], 'top': final_box[i][0][1], 'width': final_box[i][0][2], 'height': final_box[i][0][3], 'title': row.lines_combine, 'detected_as': final_box[i][1]}, ignore_index=True)
                text = row.lines_combine
                char = ['/',':','*','?','"','<','>','|']
                for c in char:
                    text = text.replace(c, "")
                final_box[i].append(text)
                get_title = True
                break
        if not get_title:
            if final_box[i][1] == 'table':
                df_image_title = df_image_title.append({'image_name': 'combined_'+str(i), 'left': final_box[i][0][0], 'top': final_box[i][0][1], 'width': final_box[i][0][2], 'height': final_box[i][0][3], 'title': 'table_'+str(count_table), 'detected_as': final_box[i][1]}, ignore_index=True)
                final_box[i].append('table_'+str(count_table))
                count_table+=1
            else:
                df_image_title = df_image_title.append({'image_name': 'combined_'+str(i), 'left': final_box[i][0][0], 'top': final_box[i][0][1], 'width': final_box[i][0][2], 'height': final_box[i][0][3], 'title': 'nontable_'+str(count_nontable), 'detected_as': final_box[i][1]}, ignore_index=True)
                final_box[i].append('nontable_'+str(count_nontable))
                count_nontable+=1
    # print(final_box)
    return(df_image_title,final_box)

def information_extraction(file_pdf, input_directory, result_folder_name, size_, pytesseract_dir, poppler_path):
    pytesseract.pytesseract.tesseract_cmd = pytesseract_dir

    os.chdir(input_directory)

    # create result_directory
    while os.path.isdir(result_folder_name):
        result_folder_name = result_folder_name + ' - Copy'
    print('Create folder name ',result_folder_name)
    os.mkdir(result_folder_name)

    result_directory = input_directory + '/' + result_folder_name
    os.chdir(result_directory)

    # find out how many pages needed in 1 pdf
    images1 = convert_from_bytes(open(file_pdf, "rb").read(), size=10, poppler_path=poppler_path)

    for i in range(len(images1)):
        # if i == 10:
        #     break
        i = 8
        start_time = timeit.default_timer()
        # convert page of i file into image
        images = convert_from_bytes(open(file_pdf, "rb").read(), size=size_, poppler_path=poppler_path, first_page=i+1, last_page=i+1)

        print('page', i)
        os.chdir(result_directory)
        os.mkdir('page '+str(i))
        os.chdir(result_directory+'/'+'page '+str(i))

        # save image of page i to png
        images[0].save('page '+str(i)+'.png', "PNG", quality=95,optimize=True, progressive=True)

        file = 'page '+str(i)+'.png'
        img_ori = cv2.imread(file)
        img_gray = cv2.imread(file,0)
        height_, width_ = img_gray.shape

        # return boundingbox of table and non-table image using hough transform
        table, nontable = get_lines(img_ori, img_gray)
        detected_table_line = table.copy()
        detected_nontable_line = nontable.copy()

        # remove table that too big within page, if there is any misdetection
        # for tab in range(len(table)):
        #     if (float(((table[tab][2]*table[tab][3])/(height_*width_))*100)) > 80:
        #         table.remove(table[tab])

        # return dataframe from each table
        # dataframe_ = create_table(table, img_gray)

        # return bounding box image within page detected using pytesseract model
        cek_tables = return_BoundingBox_Pytesseract(img_ori)
        detected_pytesseract = cek_tables.copy()

        # return final df of text within page
        _, list_df_text_grouped, df_combined_final = return_text_non_table(img_ori, img_gray, height_, width_, table, nontable)

        # remove bounding box image that too big within page, if there is any misdetection
        # for tab in range(len(cek_tables)):
        #     if (float(((cek_tables[tab][2]*cek_tables[tab][3])/(height_*width_))*100)) > 40:
        #         cek_tables.remove(cek_tables[tab])

        # algorithm to combine 2 bounding box from hough transform and pytesseract model, into final images
        final_tables, _ = combined_tables_and_nontable(table, nontable, cek_tables, height_, width_)
        final_tables2, dict1 = combined_tables_and_nontable(table, final_tables, [], height_, width_)
        combined_tables = final_tables2.copy()

        # was for debug
        if len(table) > 0:
            for i2 in range(len(table)):
                for j2 in range(len(table[i2])):
                    table[i2][j2] = int(table[i2][j2])
        if len(nontable) > 0:
            for i2 in range(len(nontable)):
                for j2 in range(len(nontable[i2])):
                    nontable[i2][j2] = int(nontable[i2][j2])
        if len(cek_tables) > 0:
            for i2 in range(len(cek_tables)):
                for j2 in range(len(cek_tables[i2])):
                    cek_tables[i2][j2] = int(cek_tables[i2][j2])
        if len(final_tables2) > 0:
            for i2 in range(len(final_tables2)):
                for j2 in range(len(final_tables2[i2])):
                    final_tables2[i2][j2] = int(final_tables2[i2][j2])

        
        # # get detected title from final_tables
        df_images_title,final_box = get_title_from_images(dict1, df_combined_final, img_ori, height_, width_)

        os.mkdir('tables')
        os.mkdir('images')

        for item in range(len(final_box)):
            if final_box[item][1] == 'table':
                os.chdir(result_directory+'/'+'page '+str(i) + '/'+'tables')
                (create_table([final_box[item][0]],img_gray,size_))[0].to_csv(str(final_box[item][2])+'.csv')
                print([final_box[item][0]])
                os.chdir(result_directory+'/'+'page '+str(i))
                save_img_table2([final_box[item][0]], img_ori, str(final_box[item][2]), i, result_directory + '/'+'page '+str(i), result_directory+'/'+'page '+str(i)+'/'+'tables')
            else:

                save_img_table2([final_box[item][0]], img_ori, str(final_box[item][2]), i, result_directory + '/'+'page '+str(i), result_directory+'/'+'page '+str(i)+'/'+'images')

        # create folder for dataframe of text within page and store csv(s)
        os.mkdir('text')
        os.chdir(result_directory+'/'+'page '+str(i)+'/'+'text')
        df_combined_final.to_csv('text.csv')
        os.chdir(result_directory+'/'+'page '+str(i))

        elapsed = timeit.default_timer() - start_time
        print('Total train time is: ',elapsed,' second(s)')
        print('Total train time is: ',elapsed/60,' minute(s)')


        #remove png file
        # os.remove(file)
        break