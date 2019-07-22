## Google Vision 

Éste proyecto tiene como objetivo el extraer datos de las credenciales de elector (IFE/INE) tanto anverso como reverso a través del uso del API de Google Vision

## Estructura

Esta desarrollado en Python v3 y para el consumo del mismo se creo una capa de serivicio REST utilizando Flask 

## Ejecución

El proyecto esta diseñado para vivir en contenedores (docker) o bien para ejecutarse de modo standalone, para ello solo se requiere ejecutar el servidor en [Flask](https://palletsprojects.com/p/flask/), que se encuentra en 

```sh
$ cd <folder>/bazteca_orc/venv/servicio
```

Y se ejecuta

```sh
$ python app.py
```

Por default se ejecuta el servicio en el puerto 5000

```sh 
 /bazteca_ocr/venv/servicio/../resources/
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 /bazteca_ocr/venv/servicio/../resources/
 * Debugger is active!
 * Debugger PIN: 137-911-863
```
