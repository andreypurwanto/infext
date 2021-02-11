from PIL import Image as Img
import os
# from ..infext.infext import information_extraction

def save_img_table2(tables, img_ori, name, page, dir_file, dir_next):
    # img_ori = cv2.imread(file)
    os.chdir(dir_next)
    # dataframe = []
    tables_img = []
    tables_img_ori = []
    for i in range(len(tables)):
        tables_img_ori.append(img_ori[tables[i][1]:tables[i][1]+tables[i][3], tables[i][0]:tables[i][0]+tables[i][2]])
        try:
            im = Img.fromarray(tables_img_ori[i])
            im.save(str(name)+".jpeg")
        except ValueError:
            pass
    os.chdir(dir_file)