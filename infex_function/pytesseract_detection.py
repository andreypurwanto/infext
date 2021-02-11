import pytesseract
from pytesseract import Output
import cv2
import pandas as pd

def pytesseract_detection(img_ori,pytesseract_dir):
    """
    Summary line.
    return the detected boundingboxes image using pytesseract.

    Parameters:
    img_ori : return from cv2.imread
    pyteserract_dir (str) : directory of pytesseract

    Return:
    tables2 : list of list of integer, contain boundingboxes of image detected with format of [[x1,y1,w1,h1],[x2,y2,w2,h2],...]
    """ 
    pytesseract.pytesseract.tesseract_cmd = pytesseract_dir
    img = img_ori
    img3 = cv2.medianBlur(img, 5)
    d2 = pytesseract.image_to_data(img3, output_type=Output.DATAFRAME)
    height_, width_, _ = img.shape
    tables2 = []
    df_tables = d2[(d2['height'] < (height_)) & (d2['height'] > (height_//15)) &
                   (d2['width'] > (width_//15)) & (d2['width'] < width_) & (d2['conf'] != -1)]
    for _, row in df_tables.iterrows():
        tables2.append([int(row.left), int(row.top),
                        int(row.width), int(row.height)])
    return(tables2)

