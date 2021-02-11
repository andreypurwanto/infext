import os
import cv2
# import numpy as np
# import pandas as pd
# import csv
# import pytesseract
# from pytesseract import Output
# import pkg_resources
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image as Img
import timeit

from infex_function.line_detection import line_detection
from infex_function.pytesseract_detection import pytesseract_detection
from infex_function.pytesseract_ocr import pytesseract_ocr
from infex_function.combination_algorithm import combination_algorithm_multiple
from infex_function.scan_title_image import scan_title_image
from infex_function.image_to_dataframe import image_to_dataframe
from infex_function.save_img import save_img_table2

def information_extraction(file_pdf, input_directory, result_folder_name, size_, pytesseract_dir, poppler_path):
    # pytesseract.pytesseract.tesseract_cmd = pytesseract_dir

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
        i = 11
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
        img_ori_copy = img_ori.copy()
        img_ori_copy2 = cv2.cvtColor(img_ori_copy, cv2.COLOR_BGR2RGB)
        img_gray = cv2.imread(file,0)
        height_, width_ = img_gray.shape
        table, nontable = line_detection(img_ori, img_gray)
        undetected_bbox = pytesseract_detection(img_ori,pytesseract_dir)
        df_combined_final = pytesseract_ocr(img_ori, img_gray, height_, width_, table, nontable, undetected_bbox, pytesseract_dir)
        dict_bbox_total_detected = combination_algorithm_multiple(table, nontable, undetected_bbox, height_, width_)
        final_box = scan_title_image(dict_bbox_total_detected, df_combined_final, img_ori, height_, width_)
        os.mkdir('tables')
        os.mkdir('images')
        
        for item in final_box:
            if final_box[item][1] == 'table':
                os.chdir(result_directory+'/'+'page '+str(i) + '/'+'tables')
                dataframe_, image_inside_table = image_to_dataframe([final_box[item][0]],img_gray,height_, width_, pytesseract_dir)
                if (len(dataframe_)>0):
                    dataframe_[0].to_csv(str(final_box[item][2])+'.csv')
                os.chdir(result_directory+'/'+'page '+str(i))
                for item2 in image_inside_table:
                    save_img_table2([image_inside_table[item2][0]], img_ori, str(image_inside_table[item2][2]), i, result_directory + '/'+'page '+str(i), result_directory+'/'+'page '+str(i)+'/'+'images')
                save_img_table2([final_box[item][0]], img_ori_copy2, str(final_box[item][2]), i, result_directory + '/'+'page '+str(i), result_directory+'/'+'page '+str(i)+'/'+'tables')
            else:
                save_img_table2([final_box[item][0]], img_ori_copy2, str(final_box[item][2]), i, result_directory + '/'+'page '+str(i), result_directory+'/'+'page '+str(i)+'/'+'images')

        # create folder for dataframe of text within page and store csv(s)
        os.mkdir('text')
        os.chdir(result_directory+'/'+'page '+str(i)+'/'+'text')
        if len(df_combined_final) > 0: 
            df_combined_final.to_csv('text.csv')
        os.chdir(result_directory+'/'+'page '+str(i))

        elapsed = timeit.default_timer() - start_time
        print('Total extracting time is: ',elapsed,' second(s)')
        print('Total extracting time is: ',elapsed/60,' minute(s)')


        #remove png file
        # os.remove(file)
        break