import os
import io
import sys
import time
import uuid
from flask import Flask, request, redirect, flash, render_template
from flask import send_from_directory
from werkzeug.utils import secure_filename

from engines import opencv


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

#Envia las imagenes que se cargan a la carpeta /resources
UPLOAD_FOLDER = os.path.join(dir_path, '../resources/')
#Envia las imagenes procesadas a la carpeta /processing
FINAL_FOLDER = dir_path + "/../processing/"
print(UPLOAD_FOLDER)
#Extensiones de imagenes que permite
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FINAL_FOLDER'] = FINAL_FOLDER
app.secret_key = 'veey3V5Vy6s45s7v57segvr'
app.config['SESSION_TYPE'] = 'filesystem'


#Revisa que la imagen tenga una extension permitida
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        s_loadtime = time.time()
        #Comprueba si se cargo el archivo
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        #Comprueba si no se selecciono el archivo
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        #Carga el archivo en la carpeta de carga
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            new_filename = str(uuid.uuid1()) + "_" + filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
            _loadtime = time.time() -s_loadtime
            start_time = time.time()
            new_file = opencv.process(app.config['UPLOAD_FOLDER'],
                                      new_filename,
                                      app.config['FINAL_FOLDER'])
            #in changes
            #datax = opencv.matchTemplate(new_file, new_filename)
            datax = opencv.process_content(new_file, new_filename)
            _time = time.time() - start_time
            _raw = datax.pop("raw", "")
            image_text = datax
            context = {
                'text': image_text,
                'src': new_filename,
                'raw': _raw,
                '_time': f"{_time:.4f} s",
                '_loadtime': f"{_loadtime:.4f} s"
            }
            #Devuelve la pagina HTML de respuesta
            return render_template("show_response.html", context=context)
    #Devuelve la pagina HTML home
    return render_template("base.html")

@app.route('/plantillador', methods=['GET', 'POST'])
def plantillador():
    """
    if request.method == 'POST':
        print(f"REQUEST: {request}")
        #Comprueba si se cargo el archivo
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        #Comprueba si no se selecciono el archivo
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        #Carga el archivo en la carpeta de carga
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # new_filename = str(uuid.uuid1()) + "_" + filename
            new_filename = "tmp.img"
            file.save(os.path.join("static/plantillador", new_filename))
        print(f"New filename: {filename}")
        return render_template("plantillador.html", _basedir = "static", _file = os.path.join("plantillador", new_filename))
    """
    return render_template("plantillador.html", _basedir = "static", _file = "plantillador/exmaple.jpg")

#Devuelve la imagen procesada
@app.route('/get/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['FINAL_FOLDER'],
                               filename)

#Devuelve la imagen sin procesar
@app.route('/get_source/<filename>')
def uploaded_file_source(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
