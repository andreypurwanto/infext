import pytesseract
from pytesseract import Output
import cv2
import pandas as pd

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

def combine_lines_and_bb(x):
    return pd.Series(dict(left=x['left'].min(), top=x['top'].min(), width=x['width'].max(), height=x['height'].sum(), lines_combine="%s" % ' '.join(x['text'])))

def combine_lines_and_bb2(x):
    return pd.Series(dict(left=x['left'].min(), top=x['top'].min(), width=x['width'].max(), height=x['height'].sum(), lines_combine="%s" % ', '.join(x['lines_combine'])))

def pytesseract_ocr(img_ori, img_gray, height_, width_, table_, nontable_, undetected_bbox, pytesseract_dir):
    """
    Summary line.
    return the string result from ocr outside images each page per line.

    Parameters:
    img_ori : return from cv2.imread
    img_gray : return from cv2.imread grayscale
    height_  (int) : of height of pixel from page
    width_ (int) : of width of pixel from page
    table_ : list of list of integer contain boundingboxes of table detected with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    nontable_ : list of list of integer contain boundingboxes of nontable detected with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    undetected_bbox : list of list of integer contain boundingboxes of bbox of images that not detected as a table or nontable with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    pyteserract_dir (str) : directory of pytesseract

    Return:
    df_combined_final : dataframe with columns (left (int):bbox left, top (int):bbox top, width(int):bbox width, height(int):bbox height, lines_combine (str): sentences from each line, detected_as (str) : detected as "text" or "title", group (int):group from each sentences)

    """ 
    pytesseract.pytesseract.tesseract_cmd = pytesseract_dir
    img = img_ori
    img2 = img_gray
    img3 = cv2.medianBlur(img, 5)
    d2 = pytesseract.image_to_data(img3, output_type=Output.DATAFRAME)
    debug_dataframe = pd.DataFrame(data=None, columns=d2.columns)
    tables = table_+nontable_+undetected_bbox
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
                first_row = False
            else:
                if(previous_x-(height_//70) < row.top < previous_x+(height_//70)):
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
                    first_row = False
                else:
                    if(previous_x-(height_//70) < row.top < previous_x+(height_//70)):
                        extracted_text[count].append(row.text)
                        previous_x = row.top
                        debug_dataframe = debug_dataframe.append(row, ignore_index=True)
                    else:
                        count = count+1
                        extracted_text.append([])
                        extracted_text[count].append(row.text)
                        previous_x = row.top
                        debug_dataframe = debug_dataframe.append(row, ignore_index=True)
    if len(debug_dataframe) > 0:
        df_combined_sentences_bb = debug_dataframe.groupby(by = ["block_num","par_num",'line_num']).apply(combine_lines_and_bb)
        df_combined_sentences_bb = df_combined_sentences_bb.reset_index()
        df_combined_sentences_bb['lines'] = (df_combined_sentences_bb.index) + 1
        df_combined_sentences_bb = df_combined_sentences_bb[['lines','left','top','width','height','lines_combine']]
        df_combined_sentences_bb = (df_combined_sentences_bb.sort_values(by=['top'])).reset_index(drop=True)
        df_combined_sentences_bb['lines_par'] = 0
        lines_par_count = 1
        bbox_text = []
        blur = cv2.GaussianBlur(img2, (7, 7), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=20)
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        count_ = []      
        for c in cnts:
            check = True
            x, y, w, h = cv2.boundingRect(c) 
            count_.append([x, y, w, h])
        df_bbox = pd.DataFrame(count_, columns=['y','x','w','h'])
        df_bbox = (df_bbox.sort_values(by=['x'])).reset_index(drop=True)
        for index1,row1 in df_bbox.iterrows() :  
            y, x, w, h = [row1.y,row1.x,row1.w,row1.h] 
            for index, row in df_combined_sentences_bb.iterrows():
                if row.lines_par == 0 and get_iou([y, x, w, h],[row['left'], row['top'], row['width'], row['height']]) > 0:
                    df_combined_sentences_bb.loc[index,'lines_par'] = lines_par_count
            lines_par_count+=1
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
            df_lines_par = df_lines_par.groupby('lowercase_tag').apply(combine_lines_and_bb2)
            df_lines_par = df_lines_par.reset_index()
            if (len(df_lines_par) == 1):
                df_lines_par['detected_as'] = 'text'
            else:
                df_lines_par['detected_as'] = 'text'
                df_lines_par.loc[0, 'detected_as'] = 'title'
            df_lines_par = df_lines_par.drop(columns='lowercase_tag')
            list_df_text_grouped.append(df_lines_par)
        for i in range(len(list_df_text_grouped)):
            list_df_text_grouped[i]['group'] = i+1
        if len(list_df_text_grouped)>0:
            df_combined_final = pd.concat(list_df_text_grouped)
        else:
            df_combined_final = pd.DataFrame(columns = ['left','top','width','height','lines_combine','detected_as','group'])
    else:
        df_combined_sentences_bb = []
        list_df_text_grouped = []
        df_combined_final = []
    return(df_combined_final)
