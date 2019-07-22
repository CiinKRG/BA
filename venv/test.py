import base64
import os
import re
import sys
import math

from googleapiclient import discovery
from googleapiclient import errors
import nltk
from nltk.stem.snowball import EnglishStemmer
from oauth2client.client import GoogleCredentials



DISCOVERY_URL = 'https://vision.googleapis.com/$discovery/rest?version=1'  # noqa
BATCH_SIZE = 1


class VisionApi:
    def __init__(self, api_discovery_file='richit-d0526bca68ad.json'):
        self.credentials = GoogleCredentials.get_application_default()
        self.service = discovery.build(
            'vision', 'v1', credentials=self.credentials,
            discoveryServiceUrl=DISCOVERY_URL)

    def detect_text(self, input_filenames, num_retries=3, max_results=6):
        images = {}
        for filename in input_filenames:
            file_name = os.path.join(os.path.dirname(__file__),'prepro/' + filename)
            with open(file_name, 'rb') as image_file:
                images[filename] = image_file.read()

        batch_request = []

        print("--------------------------")
        for filename in images:
            batch_request.append({
                'image': {
                    'content': base64.b64encode(
                            images[filename]).decode('UTF-8')
                },
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': max_results,
                }]
            })
        request = self.service.images().annotate(
            body={'requests': batch_request})

        try:
            print(" trying execute")
            responses = request.execute(num_retries=num_retries)
            
            if 'responses' not in responses:
                return {}
            text_response = {}
            for filename, response in zip(images, responses['responses']):
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
                    text_response[filename] = response['textAnnotations']

                else:
                    text_response[filename] = []
            return text_response
        except errors.HttpError as e:
            print("Http Error for %s: %s" % (filename, e))
        except KeyError as e2:
            print("Key error: %s" % e2)


def distance(p0, p1): 
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2) 

def cleaner(dirtyDict):
    if dirtyDict:
        dirtyDict["nombre"] = dirtyDict["nombre"].replace("NOMBRE", "").strip()
        dirtyDict["direccion"] = dirtyDict["direccion"].replace("DIRECCION", "").strip()
        dirtyDict["sexo"] = dirtyDict["sexo"].replace("SEXO", "").strip()
        dirtyDict["anio_registro"] = dirtyDict["anio_registro"].strip()
    return dirtyDict


def parseData(ineData):        
    lstName = []
    lstAddr = []
    dictData = {"nombre":"", "direccion":"", "cp":"", "sexo":"", "fecha_nacimiento":"", "anio_registro":"", "curp":"","registro_elector":"", "estado":"", "municipio":"", "seccion":"", "localidad":"", "vigencia":""}

    for d in ineData:
        cord = d['boundingPoly']['vertices']

        x1,y1,x2,y2,x3,y3,x4,y4 = (int(cord[0]["x"]), int(cord[0]["y"]), int(cord[1]["x"]),int(cord[1]["y"]),int(cord[2]["x"]),int(cord[2]["y"]),int(cord[3]["x"]),int(cord[3]["y"]))

        xc = ((x2 - x1) / 2) + x1
        yc = ((y4 - y1) / 2) + y1
            
        if (xc > 300 and xc < 450) and (yc > 190 and yc < 280):
            lstName.append(d['description'].upper())
            dictData["nombre"] = dictData["nombre"] + " " + d['description'].upper()

        elif (xc > 280 and xc < 850) and (yc > 320 and yc < 388):
            lstAddr.append(d['description'].upper())
            dictData["direccion"] = dictData["direccion"] + " " + d['description'].upper()

        elif (xc > 730 and xc < 842) and (yc > 218 and yc < 264):
            dictData["sexo"] =  d['description'].upper()

        elif (xc > 740 and xc < 860) and (yc > 410 and yc < 450):
            dictData["anio_registro"] = dictData["anio_registro"]  + " " + d['description']

        elif (xc > 526 and xc < 582) and (yc > 384 and yc < 416):
            dictData["registro_elector"] = d['description']

        elif (xc > 430 and xc < 487) and (yc > 414 and yc < 446):
            m = re.search("[A-Z]{1}[AEIOU]{1}[A-Z]{2}[0-9]{2}(0[1-9]|1[0-2])(0[1-9]|1[0-9]|2[0-9]|3[0-1])[HM]{1}(AS|BC|BS|CC|CS|CH|CL|CM|DF|DG|GT|GR|HG|JC|MC|MN|MS|NT|NL|OC|PL|QT|QR|SP|SL|SR|TC|TS|TL|VZ|YN|ZS|NE)[B-DF-HJ-NP-TV-Z]{3}[0-9A-Z]{1}[0-9]{1}", d['description'].upper())
            if m:
                dictData["curp"] = m.group(0)

        elif (xc > 355 and xc < 408) and (yc > 449 and yc < 488):
            dictData["estado"] = d['description']

        elif (xc > 506 and xc < 598) and (yc > 453 and yc < 489):
            dictData["municipio"] = d['description']

        elif (xc > 678 and xc < 758) and (yc > 449 and yc < 489):
            dictData["seccion"] = d['description']

        elif (xc > 678 and xc < 758) and (yc > 480 and yc < 526):
            dictData["vigencia"] = d['description']

        elif (xc > 329 and xc < 448) and (yc > 482 and yc < 525):
            dictData["localidad"] = d['description']

        elif (xc > 541 and xc < 606) and (yc > 482 and yc < 525):
            dictData["emision"] = d['description']
        else:
            m = re.search('(\d+\/\d+\/\d+)',d['description'])
            if m:
                dictData["fecha_nacimiento"] = m.group(0)
            m = re.search('(?:0[1-9]\d{3}|[1-4]\d{4}|5[0-2]\d{3})',d['description'])
            if m:
                dictData["cp"] = m.group(0)


    return dictData = cleaner(dictData)
    
       

#with open(os.path.join(os.path.dirname(__file__),'resulter.txt'), 'a') as the_file:
#the_file.write('{0} - {1} - ({2},{3})~ \n'.format(cord, d['description'].encode('utf-8'), str(xc),str(yc)))

''' Get list of images to process
'''
lstFiles = os.listdir(os.path.join(os.path.dirname(__file__),'prepro'))

''' Instance ofr VisionApi
'''
vs = VisionApi('/Users/axelgr/Documents/RichIT/Proyectos/Google/richit-d0526bca68ad.json')

''' Pass list of image and detect via api text
'''
contentOCR = vs.detect_text(lstFiles)

''' Iter all images text 
'''
for k, v in contentOCR.items():
    print(k)
    print(parseData(v))

print("Done!!")





