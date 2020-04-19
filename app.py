from flask import Flask, flash, request, redirect, url_for
from flask_cors import CORS, cross_origin
import numpy as np
import keras.models
from keras.preprocessing import image
import sys
import os
from loadModel import * 

app = Flask(__name__)

CORS(app)
#app.debug = True
app.url_map.strict_slashes = False
#global vars for easy reusability
global loaded_model, graph, session
#initialize these variables
loaded_model, graph, session = init()


@app.route('/predict/',methods=['GET', 'POST'])
def predict():

	# GET returns result for hardcoded image
	if request.method == 'GET':
		print("hello")
		img_pred=image.load_img('./DATA/Validation/NONCovid/5.jpeg',target_size=(150,150))
		img_pred=image.img_to_array(img_pred)
		img_pred=np.expand_dims(img_pred,axis=0)
		with session.as_default():
			with graph.as_default():
				out = loaded_model.predict(img_pred)
				print(out)
				if out[0][0]==1:
					prediction = "NONCOVID"
				else:
					prediction = "COVID"
				ress = str(out) +" : prediction -> "+str(prediction)
				return ress

	# POST route
	if request.method == 'POST':
		if 'file' not in request.files:
			print("no file")
			return "no file"
		file = request.files["file"]
		print(file.filename)
		if file.filename == '':
			print("no file")
			return "no file"
		else:
			print("something")
			filename = file.filename
			file.save(os.path.join("./uploads", filename))

		img_pred=image.load_img(os.path.join("./uploads", filename),target_size=(150,150))
		img_pred=image.img_to_array(img_pred)
		img_pred=np.expand_dims(img_pred,axis=0)
		with session.as_default():
			with graph.as_default():
				out = loaded_model.predict(img_pred)
				print(out)
				if out[0][0]==1:
					prediction = "NONCOVID"
				else:
					prediction = "COVID"
				ress = str(out) +" : prediction -> "+str(prediction)
				return ress


if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5001))
	app.run(host='0.0.0.0', port=port)
