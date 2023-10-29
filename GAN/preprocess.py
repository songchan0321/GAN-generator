# -*- coding: utf-8 -*-


from PIL import Image, ImageDraw, ImageFont
import os
from skimage.transform import rotate, AffineTransform, warp
import glob
import numpy as np

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32


file_path_gen = "./Generated_handwrites/*.jpg"
file_path_hnd = "./Writed_handwrites/*.jpg"
file_path = "./preprocessed_img/"
file_csv = "./input_data.csv"

files_gen = glob.glob(file_path_gen)
files_hnd = glob.glob(file_path_hnd)
files = files_gen + files_hnd

	
csv_file = open(file_csv, "w")
total = 0
for f in files:
    total += 1
    img = Image.open(f).convert("L")
    img = np.array(img)
    
    left_image = rotate(img, angle=15, cval=1)  
    right_image1 = rotate(img, angle=-15, cval=1) 
    	
    hori_transform = AffineTransform(translation=(27,0))  
    warp_r_image = warp(img, hori_transform, mode="wrap")
    	
    verti_transform = AffineTransform(translation=(0,27))  
    warp_l_image = warp(img, verti_transform, mode="wrap")
    
    augmented_img = [left_image, right_image1, warp_r_image, warp_l_image]
    
    for i in range(len(augmented_img)):
        img = augmented_img[i] * 255
        img = img.astype(int)
        arr = np.concatenate(img , axis = 0)
        arr = ",".join([str(x) for x in arr])
        csv_file.write(arr + "\n")
        img = Image.fromarray(img).convert("RGB")
        img.save(os.path.join(file_path, str(total)+"_"+str(i) + ".jpg"), 'JPEG')
    
csv_file.close()
    