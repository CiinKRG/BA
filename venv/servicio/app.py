import os
import io
import sys
import uuid
from flask import Flask, request, redirect, flash, render_template
from flask import send_from_directory
from werkzeug.utils import secure_filename

from engines import opencv


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)


UPLOAD_FOLDER = os.path.join(dir_path, '../resources/')
FINAL_FOLDER = dir_path + "/../processing/"
print(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FINAL_FOLDER'] = FINAL_FOLDER
app.secret_key = 'veey3V5Vy6s45s7v57segvr'
app.config['SESSION_TYPE'] = 'filesystem'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            new_filename = str(uuid.uuid1()) + "_" + filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
            new_file = opencv.process(app.config['UPLOAD_FOLDER'],
                                      new_filename,
                                      app.config['FINAL_FOLDER'])
            #in changes
            #datax = opencv.matchTemplate(new_file, new_filename)
            datax = opencv.process_content(new_file, new_filename)
           
            image_text = datax
            context = {
                'text': image_text,
                'src': new_filename
            }
            return render_template("show_response.html", context=context)
    return render_template("base.html")


@app.route('/get/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['FINAL_FOLDER'],
                               filename)


@app.route('/get_source/<filename>')
def uploaded_file_source(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
