import pandas as pd

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

def scan_title_image(final_box, df_combined_sentences_bb, img_ori, height_, width_):
    """
    Summary line.
    scan possible title from each of images (table and non-table) by creating an imaginary box outside of each image.

    Parameters:
    image_inside_table (dict) : dictionary with format {count:[[boundingBoxes],string of 'non_table' or 'table']} 
    tables : list of list of boundingboxes of images of tables with [[x1,y1,w1,h1],[x2,y2,w2,h2],...], ex : [[1485, 2843, 3768, 3773],[10, 20, 1000, 50]]
    img_ori : return from cv2.imread
    height_  (int) : of height of pixel from page
    width_ (int) : of width of pixel from page

    Return:
    final_box (dict) : dictionary with format {count:[[boundingBoxes],string of 'non_table' or 'table',detected title]} 
    """ 
    df_image_title = pd.DataFrame(columns=['image_name', 'left', 'top', 'width', 'height', 'title', 'detected_as'])
    divider = 18
    temp = [0,0,0,0]
    count_table = 0
    count_nontable = 0
    for i in range(len(final_box)):
        get_title = False
        if (final_box[i][0][0] - (height_//divider)) < 0:
            temp[0] = 0
        else:
            temp[0] = final_box[i][0][0] - (height_//divider)
        if (final_box[i][0][1] - (width_//divider)) < 0:
            temp[1] = 0
        else:
            temp[1] = final_box[i][0][1] - (height_//divider)
        temp[2] = final_box[i][0][2] + (width_//divider)*1.8
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
    return(final_box)