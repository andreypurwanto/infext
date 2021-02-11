import cv2
import pandas as pd
import numpy as np
import pytesseract
from pytesseract import Output

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

def combine_text(x):
    return pd.Series(dict(text="%s" % ' '.join(x['text'])))


def image_to_dataframe(tables, img_gray, height_, width_,pytesseract_dir):
    """
    Summary line.
    convert images of table to list of dataframes and bounding box of scanned images from each table

    Parameters:
    tables : list of list of boundingboxes of images of tables with [[x1,y1,w1,h1],[x2,y2,w2,h2],...], ex : [[1485, 2843, 3768, 3773],][10, 20, 1000, 50]
    img_gray : return from cv2.imread with grayscale
    height_  (int) : of height of pixel from page
    width_ (int) : of width of pixel from page
    pyteserract_dir (str) : directory of pytesseract

    Return:
    dataframe : list of dataframe from each images of tables
    image_inside_table (dict) : dictionary with format {count:[[boundingBoxes],string of 'non_table' or 'table',detected title]} 
    """ 
    
    pytesseract.pytesseract.tesseract_cmd = pytesseract_dir
    img = img_gray
    dataframe = []
    tables_img = []
    image_inside_table = {}
    #push each of the pixels of the image of tables to the list of tables_img
    for i in range(len(tables)):
        tables_img.append(img[tables[i][1]:tables[i][1]+tables[i][3], tables[i][0]:tables[i][0]+tables[i][2]])
    #iterate each of the image of tables
    for x_image in range(len(tables_img)):
        try:
            img = tables_img[x_image]
            thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # inverting the image
            img_bin = 255-img_bin
            # countcol(width) of kernel as 100th of total width
            kernel_len = np.array(img).shape[1]//25
            # Defining a vertical kernel to detect all vertical lines of image
            ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
            # Defining a horizontal kernel to detect all horizontal lines of image
            hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
            # A kernel of 2x2
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            # Use vertical kernel to detect and save the vertical lines in a jpg
            image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
            vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
            # Use horizontal kernel to detect and save the horizontal lines in a jpg
            image_2 = cv2.erode(img_bin, hor_kernel, iterations=5)
            horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=5)
            # Combine horizontal and vertical lines in a new third image, with both having same weight.
            img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
            # Eroding and thresholding the image
            img_vh = cv2.erode(~img_vh, kernel, iterations=2)
            thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # cv2.imwrite("img_vh.jpg", img_vh)
            bitxor = cv2.bitwise_xor(img, img_vh)
            bitnot = cv2.bitwise_not(bitxor)
            # Detect contours for following box detection
            contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Sort all the contours by top to bottom.
            contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
            # Creating a list of heights for all detected boxes
            heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
            # Get mean of heights
            mean = np.mean(heights)
            # Create list box to store all boxes in
            box = []
            # Get position (x,y), width and height for every contour and show the contour on image
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if (w < tables[x_image][2]//1.1 or h < tables[x_image][3]//1.1) and w > width_//500 and h > height_//500 :
                    image = cv2.rectangle(
                        img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    box.append([x, y, w, h])
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
                if countcol < len(row[i]):
                    countcol = len(row[i])
            # Retrieving the center of each column
            center = [int(row[i][j][0]+row[i][j][2]/2)
                      for j in range(len(row[i])) if row[0]]
            center = np.array(center)
            center.sort()
            # Regarding the distance to the columns center, the boxes are arranged in respective order
            finalboxes = []
            for i in range(len(row)):
                lis = []
                for k in range(countcol):
                    lis.append([])
                last_indexing = countcol
                for j in range(len(row[i])):
                    if len(row[i])==countcol:
                        diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
                        minimum = min(diff)
                        indexing = list(diff).index(minimum)
                        lis[indexing].append(row[i][j])
                    else:
                        diff = abs(center-(row[i][j][0]))
                        minimum = min(diff)
                        indexing = list(diff).index(minimum)
                        for l in range(indexing,last_indexing,1):
                            lis[l].append(row[i][j])
                        last_indexing = indexing
                finalboxes.append(lis)
            # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
            outer = []
            count_row = []
            count_row_ = 0
            check = 0
            image_inside_table = {}
            for i in range(len(finalboxes)):
                for j in range(len(finalboxes[i])):
                    count_row_ = count_row_+1
                    inner = ''
                    if(len(finalboxes[i][j]) == 0):
                        outer.append(' ')
                    else:
                        for k in range(len(finalboxes[i][j])):
                            #get the image of each cell
                            y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], finalboxes[i][j][k][3]
                            finalimg = bitnot[x:x+h, y:y+w]
                            #get the contours of cell image to detect wether the image is flip horizontally or not
                            blur = cv2.GaussianBlur(finalimg, (7, 7), 0)
                            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                            dilate = cv2.dilate(thresh, kernel, iterations=20)
                            cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                            hmax = 0
                            wmax = 0
                            for c in cnts:
                                check = True
                                x1, y1, w1, h1 = cv2.boundingRect(c)
                                if h1 > hmax:
                                  hmax = h1
                                if w1 > wmax:
                                  wmax = w1
                            #the condition if image is fliped horizontally
                            if int(wmax*1.1)<hmax:
                                finalimg = cv2.rotate(finalimg, cv2.cv2.ROTATE_90_CLOCKWISE)
                            #ocr with pytesseract
                            d2 = pytesseract.image_to_data(finalimg, output_type=Output.DATAFRAME, config='-l eng+gcr --psm 6')
                            d2 = d2.dropna()
                            #find the possible image within the cell
                            df_contain_image = d2[(d2['width']*d2['height']>(height_*width_)//420) & (d2['conf']!= -1)]
                            #push the possible image to dictionary
                            if len(df_contain_image) > 0:
                                for idx,row_df_contain_image in df_contain_image.iterrows():
                                    if row_df_contain_image.width > 0 and row_df_contain_image.height > 0:
                                        image_inside_table["ist"+str(x_image)+"-"+str(k)+"-"+str(j)+"-"+str(i)+"-"+str(idx)] = [[tables[x_image][0]+int(row_df_contain_image.left+y),tables[x_image][1]+int(row_df_contain_image.top+x),int(row_df_contain_image.width),int(row_df_contain_image.height)],'non-table',"imagefromtable"+str(x_image)+"-"+str(k)+"-"+str(j)+"-"+str(i)+"-"+str(idx)] 
                                d2 = d2.drop(index=df_contain_image.index)
                            #change the type of ocr into str
                            d2["text"]= d2["text"].values.astype(str)
                            #because we use output as a dataframe, ocr will do per words, so need to combine words for every line
                            d2 = d2.groupby(by = ["par_num",'line_num']).apply(combine_text)
                            d2 = d2.reset_index()
                            #combine every line from the result of ocr
                            out = ''
                            for index_df,row_df in d2.iterrows():
                                if index_df == 0:
                                    out += row_df.text
                                else :
                                    out = out + '\n' + row_df.text 
                            #special case if the ocr result is less than 3 character
                            if(len(out) < 3):
                                out = pytesseract.image_to_string(
                                    finalimg, config='-l eng+gcr --psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
                                if(len(out)==0):
                                    out = ' '
                            inner = inner + " " + out
                        check = check+1
                        outer.append(inner)
            count_row.append(count_row_)
            arr = np.array(outer)
            dataframe.append(pd.DataFrame(arr.reshape(len(row), countcol)))
        except IndexError:
            pass
    return(dataframe,image_inside_table)
