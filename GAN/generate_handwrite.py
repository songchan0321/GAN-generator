
from PIL import Image, ImageDraw, ImageFont
import os

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

duplicate = 30

import glob
fonts = glob.glob("./fonts/*.ttf")
file_path ="./Generated_handwrites/"

total = 0
for dup in range(duplicate):
    for character in ["", "", "", "", ""]:  
        for font in fonts:
            total += 1
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color = 255)
            drawing = ImageDraw.Draw(image)
            font = ImageFont.truetype(font,32)
            w, h = drawing.textsize(character, font=font)
            
            
            drawing.text(
                ((IMAGE_WIDTH - w)/2, (IMAGE_HEIGHT - h)/2),
                character,
                fill=(0),
                font=font
                )
            image.save(os.path.join(file_path, str(total) + ".jpg"), 'JPEG')
        
    
    


