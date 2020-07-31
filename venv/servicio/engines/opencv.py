import imutils
import cv2

import base64
import os
import io
import re
import sys
import math
import numpy as np

from googleapiclient import discovery
from googleapiclient import errors
from google.cloud import vision
import nltk
from nltk.stem.snowball import EnglishStemmer
from oauth2client.client import GoogleCredentials

DISCOVERY_URL = 'https://vision.googleapis.com/$discovery/rest?version=1'  # noqa
BATCH_SIZE = 1

INLINE_REGEX = {
    "municipio": '(MUNICIPIO)(\d+|\ +)(\d+)',
    "seccion": '(SECCI\wN)(\d+|\ +)(\d+)',
    "estado": '(ESTADO)(\d+|\ +)(\d+)',
    "sexo": '(SEXO)(H|M|\ +)(H|M)?',
    "anio_registro": '(A\wO DE REGISTRO)(\ +)(\d{4})',
}

#Busca rostros en la imagen
def detect_faces(path):
    """Detects faces in an image."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    print("faces")
    print(faces)
    if len(faces) > 0:
        return True
    else:
        return False

#Busca si es la parte frontal, busca un rostro
def isFrontal(path):
    isfront = False
    num_retry = 4
    while num_retry != 0:
        if not detect_faces(path):
            print(path)
            image = cv2.imread(path)
            image = imutils.rotate_bound(image, 90)
            cv2.imwrite(path, image)
            num_retry = num_retry - 1
        else:
            isfront = True
            break
    
    return isfront

def detect_text_singlefile(pathFile, filename, num_retries=3, max_results=6):
            #Se asigna la imagen
            image = ''

            input_filename = pathFile

            with open(input_filename, 'rb') as image_file:
                image = image_file.read()

            batch_request = []

            batch_request.append({
                'image': {
                    'content': base64.b64encode(image).decode('UTF-8')
                },
                'features': [{
                    'type': 'DOCUMENT_TEXT_DETECTION',
                    'maxResults': max_results,
                }]
            })

            #Credenciales 
            credentials = GoogleCredentials.get_application_default()
            service = discovery.build('vision', 'v1', credentials=  credentials,discoveryServiceUrl=DISCOVERY_URL)
            request = service.images().annotate(
                body={'requests': batch_request})

            try:
                print(" trying execute")
                responses = request.execute(num_retries=num_retries)
                
                if 'responses' not in responses:
                    return {}
                text_response = {}
                for filename, response in zip(image, responses['responses']):
                    if 'error' in response:
                        print("API Error for %s: %s" % (
                                filename,
                                response['error']['message']
                                if 'message' in response['error']
                                else ''))
                        continue
                    
                    #if 'fullTextAnnotation' in response:
                    #    text_response[filename] = response['fullTextAnnotation']

                    if 'textAnnotations' in response:
                        text_response['textAnnotations'] = response['textAnnotations']
                        if 'fullTextAnnotation' in response:
                            text_response['fullTextAnnotation'] = response['fullTextAnnotation']

                    else:
                        text_response['fullTextAnnotation'] = []

                return text_response
            except errors.HttpError as e:
                print("Http Error for %s: %s" % (filename, e))
            except KeyError as e2:
                print("Key error: %s" % e2)

def get_zlm(pathfile, filename):
        #image = cv2.imread(os.path.join(pathfile,filename))
        image = cv2.imread(pathfile)
        #Estructura rectangular
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 28))
        #Redimensiona la imagen y la pasa a gris
        image = cv2.resize(image, (1045, 747))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Suaviza la imagen
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        #Resalta el area mas oscura de la imagen
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        #Detecta los bordes
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        #Quita la "erosion" en las imagenes o letras
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        #Marca el limite de los colores
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        #Bordes en caso de que fueran seleccionados se retiran
        p = int(image.shape[1] * 0.05)
        thresh[:, 0:p] = 0
        thresh[:, image.shape[1] - p:] = 0
        #Encuentra los contornos y los ordena
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        roi = None
        reverse = False
        #Contornos
        for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                ar = w / float(h)
                crWidth = w / float(gray.shape[1])
                #Verifica si la relacion de aspecto y el ancho estan dentro
                #Dibuja el cuadro delimitador
                if ar > 5 and crWidth > 0.75:
                        pX = int((x + w) * 0.03)
                        pY = int((y + h) * 0.03)
                        (x, y) = (x - pX, y - pY)
                        (w, h) = (w + (pX * 2), h + (pY * 2))
                        roi = image[y:y + h, x:x + w].copy()
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        break

        #Encuentra un arreglo n-dimensional
        #Regresa True o False dependiendo si es el reverso
        if isinstance(roi, np.ndarray):
                reverse = True
                pathfile = pathfile.replace(filename, 'zlm_' + filename)
                cv2.imwrite(pathfile, roi)

        return reverse

#Devuelve el reverso del ife        
def get_ife_reverse(pathfile, filename):
        image = cv2.imread(pathfile)

        #Estructura rectangular
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        #image = cv2.resize(image, (1045, 747))
        #Se cambia a blanco y negro
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Suaviza la imagen
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        #Resalta el area mas oscura de la imagen
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        #Detecta los bordes
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        #Quita la erosion en imagenes o letras
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        #Marca el limite de los colores
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        #Quita ruido blanco de la imagen
        thresh = cv2.erode(thresh, None, iterations=4)
        #Quita los bordes
        p = int(image.shape[1] * 0.05)

        thresh[:, 0:p] = 0
        thresh[:, image.shape[1] - p:] = 0
        #Encuentra los contornos y los ordena
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        ife_reverse = False

        for c in cnts:
                #Asigna el perimetro de la curva y dibuja el contorno
                epsilon = cv2.arcLength(c,True)
                approx = cv2.approxPolyDP(c,epsilon * 0.01,True)
                #Aqui asigna si es el reverso de un ife
                if epsilon > 1200 and epsilon < 1800:   
                        #Calcula los bordes y dibuja un rectangulo delimitador
                        (x, y, w, h) = cv2.boundingRect(approx)
                        ar = w / float(h)
                        roi = image[y:y + h + 20, x:x + w].copy()
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        ife_reverse = True
                        break
                
        if ife_reverse:
                #Rota la imagen
                rotate = imutils.rotate_bound(roi, 90)
                pathfile = pathfile.replace(filename,'ifer_' + filename)
                cv2.imwrite(pathfile, rotate)

        return ife_reverse

def get_ocr_reverse(pathfile, filename):
        data = {}
        #Manda llamar la funcion que nos indica si es reverso o no
        is_reverse = get_zlm(pathfile, filename)

        #Aplica para el reverso, manda llamar a otras funciones y extrae el texto 
        if is_reverse:
                resp = visionap.detect_text_singlefile(pathfile, 'zlm_' + filename)

                if resp:
                        #print(resp)
                        dta = resp["textAnnotations"]

                        if dta:
                                valores = dta[0]['description']
                                if '\n' in valores:
                                        aux = valores.split('\n')
                                        i = 1
                                        for v in aux:
                                                data["Linea" + str(i)] = v.replace("<<", " ").replace("<" , " ")
                                                i = i + 1
        #Si la variable is_reverse es FALSE       
        #El texto que encuentre buscara 5 caracteres   
        else:
                is_reverse = get_ife_reverse(pathfile, filename) 

                if is_reverse:
                        resp = visionap.detect_text_singlefile(pathfile, 'ifer_' + filename)

                        if resp:
                                res = resp['textAnnotations']

                                for d in res:
                                        decr = d['description']
                                        m = re.search('(\d{5,})', decr)

                                        if m:
                                                data['Num'] = m.group(0)
                                                break
                                
        return data

def process(_path, filename, _target_path):
    _dest_path = os.path.join(_target_path, filename)
    image_path = _path + filename
    #rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    # load the image, resize it, and convert it to grayscale
    #Lee la imagen
    image = cv2.imread(image_path)
    
    # image = cv2.resize(image, (1045, 747))
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Guarda la imagen
    cv2.imwrite(_dest_path, image)
    #Espera indefinidamente un golpe de alguna tecla
    cv2.waitKey(0)
    return _dest_path

def get_ocr(img, filename):
    num_retries = 3
    max_results=6
    #Construye un recurso para interactuar con la API
    service = discovery.build(
            'vision', 'v1', credentials=GoogleCredentials.get_application_default(),
            discoveryServiceUrl=DISCOVERY_URL)    

    batch_request = []

    batch_request.append({
                'image': {
                    'content': base64.b64encode(img).decode('UTF-8')
                },
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': max_results,
                }]
            })
    
    #Devuelve un JSON de la imagen
    request = service.images().annotate(body={'requests': batch_request})

    try:
        #Se intentara tres veces si no se devuelve vacio para el texto encontrado aun diccionario
        responses = request.execute(num_retries=num_retries)    
        if 'responses' not in responses:
                return {}
        text_response = {}

        if responses:
                if 'error' in responses:
                        print("API Error for %s: %s" % (img,responses['error']['message']if 'message' in responses['error'] else ''))
        
        #if 'fullTextAnnotation' in response:
        #text_response[filename] = response['fullTextAnnotation']
                #Devuelve el texto encontrado en text_response
                if 'textAnnotations' in responses["responses"][0]:
                        text_response[filename] = responses["responses"][0]["textAnnotations"]
                        if 'fullTextAnnotation' in responses["responses"][0]:
                            text_response['all_' + filename] = responses["responses"][0]['fullTextAnnotation']

                else:
                        text_response[filename] = []
                        text_response['all_' + filename] = []
                            
        return text_response
    except errors.HttpError as e:
        print("Http Error for: %s" % (e))
    except KeyError as e2:
        print("Key error: %s" % e2)

#Busca la informacion de la cara frontal
def parseInfo(ineData, filename, xx1, yy1, xx3, yy3, keyword=None):
    keyWords = ["NOMBRE", "DOMICILIO", "FOLIO", "CLAVE", "LOCALIDAD", "SEXO", "ELECTOR"]

    #Busca una de las palabras
    if keyword in INLINE_REGEX:
        print("searching... ", keyword)
        lstData = search_regex(INLINE_REGEX[keyword],
                               ineData['all_' + filename]['text'])
        print(lstData)
        if lstData:
            return lstData
    #Vacia la lista
    lstData = []

    for d in ineData[filename]:
    #for d in ineData:
        #Cada texto en ineData
        cord = d['boundingPoly']['vertices']
        #Asigna las coordenadas
        x1,y1,x2,y2,x3,y3,x4,y4 = (int(cord[0]["x"]), int(cord[0]["y"]), int(cord[1]["x"]),int(cord[1]["y"]),int(cord[2]["x"]),int(cord[2]["y"]),int(cord[3]["x"]),int(cord[3]["y"]))

        xc = ((x2 - x1) / 2) + x1
        yc = ((y4 - y1) / 2) + y1
            
        #Si coincide con las coordenadas; si esta en mayusculas y no pertenece a las keywords, se agrega a la lista
        if (x1 > xx1 and y1 > yy1) and (x3 < xx3 and y3 < yy3):
            if d["description"].upper() not in keyWords:
                lstData.append(d["description"])

        #with open(os.path.join(os.path.dirname(__file__),'..','resulter.txt'), 'a') as the_file:
        #    the_file.write('{0} - {1} - ({2},{3})~ \n'.format(cord, d['description'].encode('utf-8'), str(xc),str(yc)))
    return lstData

def compare(imgPath, imgtemp):
    img = cv2.imread(imgPath,0)
    #Suaviza la imagen
    img = cv2.bilateralFilter(img,9,75,75)
    img2 = img.copy()
    #img, img2 son la imagen suavizada

    #Lee la imagen temporal
    template = cv2.imread(imgtemp,0)
    #Redimensiona la imagen
    w, h = template.shape[::-1]

    #Compara la imagen suavizada y la imagen redimensionada, y encuentra la cara
    res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    #Encuentra el minimo y maximo valor
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    #Regresa las coordenadas de la imagen
    return {"x1":top_left[0], "y1": top_left[1], "x2": top_left[0] + w, "y2": top_left[1] + h}

#Busca el texto con expresion regular
def search_regex(reg_exp, text, keep_value=False):
    sexo_regex = re.search(reg_exp, text.upper())
    if sexo_regex:
        return [sexo_regex.group(0)]
    elif keep_value:
        return text
    return ""
       
#Asigna los campos de la parte frontal a un diccionario
def cleaner(dirtyDict):
    if dirtyDict:
        
        #Separa el nombre en apellidos y nombre
        if len(dirtyDict["nombre"]) >= 3:
            dirtyDict["apellido_paterno"] = dirtyDict["nombre"][0]
            dirtyDict["apellido_materno"] = dirtyDict["nombre"][1]
            dirtyDict["nombre"] = ' '.join(dirtyDict["nombre"]).replace(dirtyDict["apellido_paterno"], "").replace(dirtyDict["apellido_materno"], "").strip()
        #Si tiene menos de 3 palabras se asigna el campo, y se reemplazan los campos de NOMBRE y DOMICILIO
        else:
            dirtyDict["nombre"] = ' '.join(dirtyDict["nombre"]).replace("NOMBRE", "").replace("DOMICILIO", "").strip()

        #Cambio en el campo de DOMICILIO
        lstDom = ' '.join(dirtyDict["direccion"]).replace("DOMICILIO", "").strip()
        dirtyDict["direccion"] = lstDom
        #Busca 5 caracteres y asigna como poblacion
        if lstDom:
            m = re.search('(\d){5}', lstDom)
            if m: 
                dirtyDict["poblacion"] = lstDom[m.span(0)[1]:].strip()

        #Asigna la calle hasta donde encuentra la palabra COL
        calle = dirtyDict["direccion"] 
        if calle.find('COL'):
            dirtyDict["calle"] = calle[0:calle.find('COL')].strip()

        #Busca 5 caracteres en la parte de direccion y asigna el CP
        cp = re.search("(\d{5})",dirtyDict["direccion"])
        if cp:
            dirtyDict["cp"] = cp.group(0)
            if dirtyDict["poblacion"]:
                dirtyDict["poblacion"] = dirtyDict["poblacion"].replace(dirtyDict["cp"], "")

        #Extraee colonia, poblacion y cp
        col = re.search("COL[\s?|\.?]([\w*|\W*]){1,}", dirtyDict["direccion"], re.I) 
        if col:
            dirtyDict["colonia"] = col.group(0).replace("COL","").replace(dirtyDict["poblacion"],"").replace(dirtyDict["cp"],"").strip()
         
        dirtyDict["sexo"] = ' '.join(dirtyDict["sexo"]).replace("SEXO", "").strip()
        dirtyDict["registro_elector"] = ' '.join(dirtyDict["registro_elector"]).replace("ELECTOR","").strip()
        dirtyDict["anio_registro"] = str(re.sub('A\wO DE REGISTRO', '', ' '.join(dirtyDict["anio_registro"]))).strip()
        dirtyDict["estado"] = ' '.join(dirtyDict["estado"]).replace("ESTADO", "").strip()
        dirtyDict["municipio"] = ' '.join(dirtyDict["municipio"]).replace("MUNICIPIO", "").strip()
        dirtyDict["localidad"] = ' '.join(dirtyDict["localidad"]).strip()
        dirtyDict["emision"] = ' '.join(dirtyDict["emision"]).strip()
        dirtyDict["seccion"] = str(re.sub('SECCI\wN', '', ' '.join(dirtyDict["seccion"]))).strip()
        dirtyDict["vigencia"] = ' '.join(dirtyDict["vigencia"]).strip()

        if 'fechaNacimiento' in dirtyDict:
            dirtyDict["fechaNacimiento"] = ' '.join(dirtyDict["fechaNacimiento"]).strip()

        dirtyDict["curp"] = ' '.join(dirtyDict["curp"]).strip()
        m = re.search("[A-Z]{1}[AEIOU]{1}[A-Z]{2}[0-9]{2}(0[1-9]|1[0-2])(0[1-9]|1[0-9]|2[0-9]|3[0-1])[HM]{1}(AS|BC|BS|CC|CS|CH|CL|CM|DF|DG|GT|GR|HG|JC|MC|MN|MS|NT|NL|OC|PL|QT|QR|SP|SL|SR|TC|TS|TL|VZ|YN|ZS|NE)[B-DF-HJ-NP-TV-Z]{3}[0-9A-Z]{1}[0-9]{1}", dirtyDict["curp"].upper())
        if m:
            dirtyDict["curp"] = m.group(0)

        if 'edad' in dirtyDict:
            dirtyDict["edad"] = ' '.join(dirtyDict["edad"]).strip()

        if 'folio' in dirtyDict:
            dirtyDict["folio"] = ' '.join(dirtyDict["folio"]).strip()

    return dirtyDict

#Coordenadas de los campos de la INE  
def matchTemplate(pathImg, ineName, dt):

    #Asigna el path de las imagenes de las palabras a encontrar en el INE
    pathNombre = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_nombre.jpg')
    pathDomicilio = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_domicilio.jpg')
    pathClaveElector = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_celector.jpg')
    pathCURP = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_curp.jpg')
    pathEstado = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_estado.jpg')
    pathLocalidad = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_localidad.jpg')
    pathMunicipio = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_municipio.jpg')
    pathEmision = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_emision.jpg')
    pathSeccion = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_seccion.jpg')
    pathVigencia = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_vigencia.jpg')
    pathFechaNacimiento = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_fechaNacimiento.jpg')
    pathRegistro = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_registro.jpg')
    pathSexo = os.path.join(os.path.dirname(__file__),'..', 'imgtemplates/ine/ine_sexo.jpg')

    p1 = compare(pathImg,pathNombre)
    p2 = compare(pathImg,pathDomicilio)
    p3 = compare(pathImg, pathClaveElector)
    p4 = compare(pathImg, pathCURP)
    p5 = compare(pathImg, pathEstado)
    p6 = compare(pathImg, pathLocalidad)
    p7 = compare(pathImg, pathMunicipio)
    p8 = compare(pathImg, pathEmision)
    p9 = compare(pathImg, pathSeccion)
    p10 = compare(pathImg, pathVigencia)
    p11 = compare(pathImg, pathFechaNacimiento)
    p12 = compare(pathImg, pathRegistro)
    p13 = compare(pathImg, pathSexo)

    dictData = {"tipo": "", "nombre":"", "direccion":"","registro_elector":"", "curp":"", "estado" : "", "municipio":"","localidad":"", "emision":"", "fechaNacimiento":"", "anio_registro":"", "sexo":""}
    #with io.open(pathImg, 'rb') as image_file:
    #    content = image_file.read()    
    #    dt = get_ocr(content, ineName)
    #Asigna las coordenadas de cada campo
    dictData["nombre"] = parseInfo(dt, ineName, int(p1['x1']), int(p1['y1']), int(p2['x2']) + 100, int(p2['y2']))
    dictData["direccion"] = parseInfo(dt, ineName, int(p2['x1']), int(p2['y1']), int(p3['x2']) + 300, int(p3['y2']) - 20)
    dictData["registro_elector"] = parseInfo(dt, ineName, int(p3['x1']) + 100, int(p3['y1'] - 15 ), int(p4['x2']) + 420, int(p4['y2']) - 35)
    dictData["curp"] = parseInfo(dt, ineName, int(p4['x1']) + 20, int(p4['y1'] - 18 ), int(p5['x2']) + 290, int(p5['y2']) - 35)
    dictData["estado"] = parseInfo(dt, ineName,
                                   int(p5['x1']) + 20, int(p5['y1'] - 18 ), int(p6['x2']) + 100, int(p6['y2']) - 35,
                                   keyword="estado")
    dictData["localidad"] = parseInfo(dt, ineName, int(p6['x1']) + 20, int(p6['y1'] - 18 ), int(p6['x2']) + 100, int(p6['y2']) + 10 )
    dictData["municipio"] = parseInfo(dt, ineName,
                                      int(p7['x1']) + 20, int(p7['y1'] - 18 ), int(p8['x2']) + 100, int(p8['y2']) - 35,
                                      keyword="municipio")
    dictData["emision"] = parseInfo(dt, ineName, int(p8['x1']) + 20, int(p8['y1'] - 18 ), int(p8['x2']) + 100, int(p8['y2']) + 10 )
    dictData["seccion"] = parseInfo(dt, ineName,
                                    int(p9['x1']) + 20, int(p9['y1'] - 18 ), int(p9['x2']) + 100, int(p9['y2']) + 35,
                                    keyword="seccion")
    dictData["vigencia"] = parseInfo(dt, ineName, int(p10['x1']) + 20, int(p10['y1'] - 18 ), int(p10['x2']) + 100, int(p10['y2']) + 10 )
    dictData["fechaNacimiento"] = parseInfo(dt, ineName, int(p11['x1']) + 10, int(p11['y1'] + 20 ), int(p11['x2']) + 100, int(p11['y2']) + 40 )
    dictData["anio_registro"] = parseInfo(dt, ineName,
                                          int(p12['x1']) + 80, int(p12['y1']) -30, int(p12['x2']) + 350 , int(p12['y2']) + 40,
                                          keyword="anio_registro")
    dictData["sexo"] = parseInfo(dt, ineName,
                                 int(p13['x1']), int(p13['y1']) - 20, int(p13['x2']) + 40, int(p13['y2']),
                                 keyword="sexo")
    print(dt['all_' + ineName]['text'])
    print(dictData)
    return cleaner(dictData)

#Coordenadas de los campos de la IFE
def matchIFETemplate(pathImg, ifeName, ocr_data):
    pathNombre = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_nombre.jpg')
    pathDomicilio = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_domicilio.jpg')
    pathFolio = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_folio.jpg')
    pathClaveElector = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_celector.jpg')
    pathCURP = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_curp.jpg')
    pathEstado = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_estado.jpg')
    pathLocalidad = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_localidad.jpg')
    pathEmision = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_emision.jpg')
    pathMunicipio = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_municipio.jpg')
    pathSeccion = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_seccion.jpg')
    pathVigencia = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_vigencia.jpg')
    pathEdad = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_edad.jpg')
    pathSexo = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_sexo.jpg')
    pathRegistro = os.path.join(os.path.dirname(__file__),'..','imgtemplates/ife/ife_registro.jpg')
    
    p1 = compare(pathImg,pathNombre)
    p2 = compare(pathImg,pathDomicilio)
    p3 = compare(pathImg, pathFolio)
    p4 = compare(pathImg, pathClaveElector)
    p5 = compare(pathImg, pathCURP)
    p6 = compare(pathImg, pathEstado)
    p7 = compare(pathImg, pathLocalidad)
    p8 = compare(pathImg, pathEmision)
    p9 = compare(pathImg, pathMunicipio)
    p10 = compare(pathImg, pathSeccion)
    p11 = compare(pathImg, pathVigencia)
    p12 = compare(pathImg, pathEdad)
    p13 = compare(pathImg, pathSexo)
    p14 = compare(pathImg, pathRegistro)

    dictData = {"tipo": "", "nombre":"", "direccion":"","registro_elector":"", "folio": "", "curp":"", "estado" : "", "municipio":"","localidad":"", "emision":"", "seccion":"", "anio_registro":"", "edad": "", "sexo":"", "vigencia": ""}
    #with io.open(pathImg, 'rb') as image_file:
    #    content = image_file.read()    
    #    dt = get_ocr(content, ifeName)
    dt = ocr_data
    dictData["nombre"] = parseInfo(dt, ifeName, int(p1['x1']), int(p1['y1']), int(p2['x2']) + 130, int(p2['y2']) -20)
    dictData["direccion"] = parseInfo(dt, ifeName, int(p2['x1']), int(p2['y1']), int(p3['x2']) + 470, int(p3['y2']) - 20)   
    dictData["folio"] = parseInfo(dt, ifeName, int(p3['x1']) + 20, int(p3['y1']) - 20, int(p4['x2']) + 90, int(p4['y2']) - 10)
    dictData["registro_elector"] = parseInfo(dt, ifeName, int(p4['x1']) + 70, int(p4['y1']) - 10, int(p5['x2']) + 450, int(p5['y2']) + 20)
    dictData["curp"] = parseInfo(dt, ifeName, int(p5['x1']) + 30, int(p5['y1'] - 18 ), int(p6['x2']) + 300, int(p6['y2']) - 20)
    dictData["estado"] = parseInfo(dt, ifeName,
                                   int(p6['x1']) + 70, int(p6['y1']) -15, int(p7['x2']) + 50, int(p7['y2']) - 10,
                                   keyword="estado")
    dictData["localidad"] = parseInfo(dt, ifeName, int(p7['x1']) + 70, int(p7['y1']) -15, int(p8['x2']) + 100, int(p8['y2']) - 10 )
    dictData["emision"] = parseInfo(dt, ifeName, int(p8['x1']) + 20, int(p8['y1'] - 18 ), int(p8['x2']) + 100, int(p8['y2']) + 10 )
    dictData["municipio"] = parseInfo(dt, ifeName,
                                      int(p6['x1']) + 200, int(p6['y1']) -15, int(p9['x2']) + 70, int(p9['y2']) + 10,
                                      keyword="municipio")
    dictData["seccion"] = parseInfo(dt, ifeName,
                                    int(p7['x1']) + 200, int(p7['y1']) -15, int(p10['x2']) + 80, int(p10['y2']) + 5,
                                    keyword="seccion")
    dictData["vigencia"] = parseInfo(dt, ifeName, int(p11['x1']) + 100, int(p11['y1'] - 18 ), int(p11['x2']) + 90, int(p11['y2']) + 10 )
    dictData["edad"] = parseInfo(dt, ifeName, int(p12['x1']) + 40, int(p12['y1']) - 15, int(p12['x2']) + 80, int(p12['y2']))
    dictData["sexo"] = parseInfo(dt, ifeName,
                                 int(p13['x1']) + 40, int(p13['y1']) - 15, int(p13['x2']) + 80, int(p13['y2']),
                                 keyword="sexo")
    dictData["anio_registro"] = parseInfo(dt, ifeName,
                                          int(p14['x1']) + 120, int(p14['y1']) - 15, int(p14['x2']) + 140, int(p14['y2']),
                                          keyword="anio_registro")
    print(dt['all_' + ifeName]['text'])
    print(dictData)
    return cleaner(dictData)

#Funcion para la parte frontal del INE
def isBackerZLM(pathImg,imgName):
    isBack = False
    num_retry = 4
    dt = {}
    while num_retry != 0:

        #Manda llamar la funcion que edita la imagen
        dt = get_zlm(pathImg, imgName)
        
        #si dt esta vacio devuelve True y entra
        if not bool(dt):
            print(pathImg)
            image = cv2.imread(pathImg)
            #Rota la imagen sin ponerle margen
            image = imutils.rotate_bound(image, 90)
            cv2.imwrite(pathImg, image)
            num_retry = num_retry - 1

        else:
            isBack = True
            break
    
    return dt

#Funcion para la parte posterior del INE
def isBackerStandard(pathImg,imgName):
    isBack = False
    num_retry = 4
    dt = {}
    while num_retry != 0:
        #Manda llamar la funcion que edita la imagen
        dt = get_ife_reverse(pathImg, imgName)
        
        if not bool(dt):
            image = cv2.imread(pathImg)
            image = imutils.rotate_bound(image, 90)
            cv2.imwrite(pathImg, image)
            num_retry = num_retry - 1

        else:
            isBack = True
            break
    
    return dt

#Extraee la informacion del reverso
def extractInfo_Reverse(pathimage,filename):
    data = {}

    #Si concuerda con la parte trasera del INE extrae el texto
    if isBackerZLM(pathimage,filename):
            pathimage = pathimage.replace(filename, "zlm_" + filename)
            print("isbackerzlm")
            print(pathimage)
            resp = detect_text_singlefile(pathimage, filename)
            dta = resp["textAnnotations"]
            data["raw"] = dta
            #Si extrae texto lo hace por linea
            if dta:
                    valores = dta[0]['description']
                    if '\n' in valores:
                            aux = valores.split('\n')
                            i = 1
                            for v in aux:
                                    data["Linea" + str(i)] = v.replace("<<", " ").replace("<" , " ")
                                    i = i + 1
    #Si no concuerda con la parte trasera del INE 
    elif isBackerStandard(pathimage,filename):
            
            print("isstandard")
            print(pathimage)
            #Extrae el texto
            resp = detect_text_singlefile(pathimage, "ifer_" + filename)

            #Busca al menor 5 caracteres y los regresa
            if resp and 'textAnnotations' in resp:
                    res = resp['textAnnotations']
                    data["raw"] = res

                    for d in res:
                            decr = d['description']
                            m = re.search('(\d{5,})', decr)

                            if m:
                                    data['Num'] = m.group(0)
                                    break

    return data

def process_content(pathImg,imgName):

    #Revisa si es la parte frontal
    isfront = isFrontal(pathImg)

    #Si es la parte frontal lee la imagen, la redimensiona, la pasa a escala de grises y la guarda
    if isfront:
        image = cv2.imread(pathImg)
        image = cv2.resize(image, (1045, 747))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(pathImg, gray)

        #Abre la imagen, la lee y manda llamar la funcion que extrae un JSON de la imagen
        #Asigna un tipo de IFE
        with io.open(pathImg, 'rb') as image_file:
            content = image_file.read()
            dt = get_ocr(content, imgName)
            typeId = 'IFE - C'
            datax = {}

            if 'all_' + imgName in dt:
                print("enter")
                result = dt['all_' + imgName]

                #Busca la palabra y asigna el tipo de INE al que pertenece
                if result['text']:
                    pretype = re.search('(\w*)\s(NACIONAL)\s(\w*)', result['text'].upper())
                    if pretype:
                        foreign_type = re.search('(\w*)\s(EXTRANJERO)\s(\w*)', result['text'].upper())
                        if foreign_type:
                            typeId = 'INE - F'
                        else:
                            etype_regex = re.search('(FECHA\s*DE\s*NACIMIENTO)|(SEXO)', result['text'].upper())
                            if etype_regex:
                                typeId = 'INE - E'
                            else:
                                typeId = 'INE - D'

            #Si es tipo IFE manda llamar la funcion de su template
            if typeId.startswith("IFE"):
                datax = matchIFETemplate(pathImg, imgName, dt)
                
            else:
                datax = matchTemplate(pathImg, imgName, dt)

            datax["raw"] = dt
            datax['tipo'] = typeId

    #Si es el reverso manda llamar la funcion que extra la info
    else:
        datax = extractInfo_Reverse(pathImg,imgName)
        
    return datax
