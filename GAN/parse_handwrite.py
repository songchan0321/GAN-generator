# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

import glob
papers = glob.glob("./papers/*.jpg")
file_path ="./Writed_handwrites/"

row = 12
col = 12

ystart = 1600
xstart = 1000

charx = 500
chary = 600
paddingx = 150
paddingy = 360



total = 0
for paper in papers:
    
    print(paper)
    image = Image.open(paper).convert("L")
    image = np.array(image)
    for r in range(row):
        for c in range(col):
            total += 1
            piece = image[ystart + (chary + paddingy) * r :ystart + (chary + paddingy) * r + chary, xstart + (charx + paddingx) * c:xstart + (charx + paddingx) * c + charx]
            print(piece.shape)
            piece = Image.fromarray(piece)
        
            
            piece = piece.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            
            piece.save(os.path.join(file_path, str(total) + ".jpg"), 'JPEG')
            





