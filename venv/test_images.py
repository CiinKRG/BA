from PIL import Image, ImageChops
import os

def trims(im):
    bg = Image.new(im.mode, im.size, im.getpixel((1,1)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

im = Image.open(os.path.join(os.path.dirname(__file__),'resources/ine_sergio.jpg'))
new_image = im.resize((1047, 747))
other = trims(new_image)
#other.show()

other.save(os.path.join(os.path.dirname(__file__),'processing/ine_sergio.jpg'))