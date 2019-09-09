from PIL import Image, ImageChops
import os

def trims(im):
    #Crea una nueva imagen
    bg = Image.new(im.mode, im.size, im.getpixel((1,1)))
    #Regresa el valor absoluto de la diferencia pixel x pixel entre dos imagenes
    diff = ImageChops.difference(im, bg)
    #Aniade dos imagenes, dividiendo el resultado por escala y agregando el offset 
    diff = ImageChops.add(diff, diff, 2.0, -100)
    #Corta los bordes 
    bbox = diff.getbbox()
    if bbox:
        #Recorta la imagen (devuelve una region rectangular)
        return im.crop(bbox)

#Une los paths
im = Image.open(os.path.join(os.path.dirname(__file__),'resources/ine_sergio.jpg'))
#Regresa una copia redimensionada
new_image = im.resize((1047, 747))
#Corta los espacios en blanco
other = trims(new_image)
#other.show()
#Une los paths y lo guarda
other.save(os.path.join(os.path.dirname(__file__),'processing/ine_sergio.jpg'))