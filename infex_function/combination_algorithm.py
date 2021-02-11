import cv2
import pandas as pd
import numpy as np
import math 
import imutils

def get_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])
    if ((x_right - x_left) < 0 and (y_bottom - y_top) < 0):
        intersection_area = (x_right - x_left) * (y_bottom - y_top) * (-1)
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]
    if(bb2_area == 0):
        iou = -1
    else:
        iou = intersection_area / float(bb2_area)
    return iou

def combination_algorithm_single(table, nontable, undetected_bbox, height_, width_):
    """
    Summary line.
    combination algorithm to combine table bbox, nontable bbox from line detection and undetected bbox from pytesseract detection.

    Parameters:
    table_ : list of list of integer contain boundingboxes of table detected with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    nontable_ : list of list of integer contain boundingboxes of nontable detected with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    undetected_bbox : list of list of integer contain boundingboxes of bbox of images that not detected as a table or nontable with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    height_  (int) : of height of pixel from page
    width_ (int) : of width of pixel from page

    Return:
    bbox_total : result of combination, could be reduction or combination of bboxes
    dict_bbox_total_detected : dictionary with format {count:[[boundingBoxes],string of 'non_table' or 'table',detected title]}
    """ 
    bbox_pytesseract = undetected_bbox
    bbox_linedetection = table+nontable
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
    return (bbox_total, dict_bbox_total_detected)

def combination_algorithm_multiple(table, nontable, undetected_bbox, height_, width_,iteration=2):
    """
    Summary line.
    iteration of combination algorithm to combine table bbox, nontable bbox from line detection and undetected bbox from pytesseract detection.

    Parameters:
    table_ : list of list of integer contain boundingboxes of table detected with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    nontable_ : list of list of integer contain boundingboxes of nontable detected with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    undetected_bbox : list of list of integer contain boundingboxes of bbox of images that not detected as a table or nontable with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    height_  (int) : of height of pixel from page
    width_ (int) : of width of pixel from page
    iteration (int) : number of iteration

    Return:
    dict_bbox_total_detected : dictionary with format {count:[[boundingBoxes],string of 'non_table' or 'table',detected title]}

    """ 
    for i in range(iteration):
        if i == 0:
            bbox_total, dict_bbox_total_detected = combination_algorithm_single(table, nontable, undetected_bbox, height_, width_)
        else :
            bbox_total, dict_bbox_total_detected = combination_algorithm_single(table, bbox_total_temp, [], height_, width_)
        bbox_total_temp = bbox_total
    return (dict_bbox_total_detected)