import os
import json
import base64
import re
from PIL import Image, ImageChops
from google.cloud import vision
from google.protobuf.json_format import MessageToJson
from googleapiclient import discovery
from googleapiclient import errors
from oauth2client.client import GoogleCredentials


client = vision.ImageAnnotatorClient()
DISCOVERY_URL = 'https://vision.googleapis.com/$discovery/rest?version=1'  # noqa

def trims(im):
    bg = Image.new(im.mode, im.size, im.getpixel((1, 1)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def process(_path, filename, _target_path):
    im = Image.open(_path + filename)
    # other = trims(im)
    other = trims(im)
#    other = other.convert('L')
    other = other.convert('1', dither=Image.NONE)
    _dest_path = os.path.join(_target_path, filename)
    other.save(_dest_path)
    return _dest_path


def get_text(img, backup_response=None):
    image = vision.types.Image(content=img)
    response = client.text_detection(image=image)
    serialized = MessageToJson(response)
    if backup_response:
        objs = json.loads(serialized)
        with open(backup_response, 'w') as fp:
            json.dump(objs, fp)
    image_text = list(map(lambda x: x.description, response.text_annotations))
    return image_text



def get_ocr(img, filename):
    num_retries = 3
    max_results=6
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
    
    request = service.images().annotate(body={'requests': batch_request})

    
    try:
        responses = request.execute(num_retries=num_retries)    
        if 'responses' not in responses:
                return {}
        text_response = {}

        if responses:
                if 'error' in responses:
                        print("API Error for %s: %s" % (img,responses['error']['message']if 'message' in responses['error'] else ''))
        
        #if 'fullTextAnnotation' in response:
        #    text_response[filename] = response['fullTextAnnotation']

                if 'textAnnotations' in responses["responses"][0]:
                        print("annotations")
                        text_response[filename] = responses["responses"][0]["textAnnotations"]
                else:
                        print("nop")
                        text_response[filename] = [responses]
        return text_response
    except errors.HttpError as e:
        print("Http Error for: %s" % (e))
    except KeyError as e2:
        print("Key error: %s" % e2)


def parseData(ineData, filename):       
    lstName = []
    lstAddr = []
    dictData = {"nombre":"", "direccion":"", "cp":"", "sexo":"", "fecha_nacimiento":"", "anio_registro":"", "curp":"","registro_elector":"", "estado":"", "municipio":"", "seccion":"", "localidad":"", "vigencia":""}

    for d in ineData[filename]:
        cord = d['boundingPoly']['vertices']

        x1,y1,x2,y2,x3,y3,x4,y4 = (int(cord[0]["x"]), int(cord[0]["y"]), int(cord[1]["x"]),int(cord[1]["y"]),int(cord[2]["x"]),int(cord[2]["y"]),int(cord[3]["x"]),int(cord[3]["y"]))

        xc = ((x2 - x1) / 2) + x1
        yc = ((y4 - y1) / 2) + y1
            

        if (xc > 300 and xc < 450) and (yc > 190 and yc < 280):
            lstName.append(d['description'].upper())
            dictData["nombre"] = dictData["nombre"] + " " + d['description'].upper()

        elif (xc > 310 and xc < 850) and (yc > 340 and yc < 440):
            lstAddr.append(d['description'].upper())
            dictData["direccion"] = dictData["direccion"] + " " + d['description'].upper()

        elif (xc > 730 and xc < 842) and (yc > 218 and yc < 264):
            dictData["sexo"] =  d['description'].upper()

        elif (xc > 820 and xc < 932) and (yc > 478 and yc < 590):
            dictData["anio_registro"] = dictData["anio_registro"]  + " " + d['description']

        elif (xc > 570 and xc < 652) and (yc > 394 and yc < 508):
            dictData["registro_elector"] = d['description']

        elif (xc > 440 and xc < 530) and (yc > 398 and yc < 540):
            m = re.search("[A-Z]{1}[AEIOU]{1}[A-Z]{2}[0-9]{2}(0[1-9]|1[0-2])(0[1-9]|1[0-9]|2[0-9]|3[0-1])[HM]{1}(AS|BC|BS|CC|CS|CH|CL|CM|DF|DG|GT|GR|HG|JC|MC|MN|MS|NT|NL|OC|PL|QT|QR|SP|SL|SR|TC|TS|TL|VZ|YN|ZS|NE)[B-DF-HJ-NP-TV-Z]{3}[0-9A-Z]{1}[0-9]{1}", d['description'].upper())
            if m:
                dictData["curp"] = m.group(0)

        elif (xc > 380 and xc < 440) and (yc > 520 and yc < 590):
            dictData["estado"] = d['description']

        elif (xc > 550 and xc < 700) and (yc > 522 and yc < 588):
            dictData["municipio"] = d['description']

        elif (xc > 740 and xc < 842) and (yc > 520 and yc < 590):
            dictData["seccion"] = d['description']

        elif (xc > 768 and xc < 850) and (yc > 560 and yc < 705):
            dictData["vigencia"] = d['description']

        elif (xc > 390 and xc < 480) and (yc > 562 and yc < 638):
            dictData["localidad"] = d['description']

        elif (xc > 560 and xc < 680) and (yc > 562 and yc < 638):
            dictData["emision"] = d['description']
        
        else:
            m = re.search('(\d+\/\d+\/\d+)',d['description'])
            if m:
                dictData["fecha_nacimiento"] = m.group(0)
            m = re.search('(?:0[1-9]\d{3}|[1-4]\d{4}|5[0-2]\d{3})',d['description'])
            if m:
                dictData["cp"] = m.group(0)


    return cleaner(dictData)


def cleaner(dirtyDict):
    if dirtyDict:
        dirtyDict["nombre"] = dirtyDict["nombre"].replace("NOMBRE", "").strip()
        dirtyDict["direccion"] = dirtyDict["direccion"].replace("DIRECCION", "").strip()
        dirtyDict["sexo"] = dirtyDict["sexo"].replace("SEXO", "").strip()
        dirtyDict["anio_registro"] = dirtyDict["anio_registro"].strip()
    return dirtyDict


def get_data(img, filename, backup_response=None):
        return parseData(get_ocr(img,filename), filename)
        
