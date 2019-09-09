import os
from PIL import Image
from image_processor import trims

def process(_path, filename, _target_path):
    im = Image.open(_path + filename)
    #Corta los espacios en blanco
    other = trims(im)
    #Copia convertida de la imagen en binivel y ninguna interpolacion
    other = other.convert('1', dither=Image.NONE)
    #Une los paths 
    _dest_path = os.path.join(_target_path, filename)
    #Guarda la imagen nueva en el path unido
    other.save(_dest_path)
    return _dest_path
