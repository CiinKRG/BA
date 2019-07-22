import os
from PIL import Image
from image_processor import trims


def process(_path, filename, _target_path):
    im = Image.open(_path + filename)
    # other = trims(im)
    other = trims(im)
#    other = other.convert('L')
    other = other.convert('1', dither=Image.NONE)
    _dest_path = os.path.join(_target_path, filename)
    other.save(_dest_path)
    return _dest_path
