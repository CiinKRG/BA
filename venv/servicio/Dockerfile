FROM python:3.7-stretch
WORKDIR /api
COPY ./secrets/ /secrets/
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/credentials_project.json
COPY ./ /api/bazteca_ocr
RUN mkdir /api/resources /api/processing
RUN apt-get update
RUN apt-get install tesseract-ocr -y
RUN apt-get install libtesseract-dev -y
RUN cd /api/bazteca_ocr && pwd && ls -la && pip install -r requirements.txt
CMD cd /api/bazteca_ocr && pwd && gunicorn --bind 0.0.0.0:5000 app:app
EXPOSE 5000
