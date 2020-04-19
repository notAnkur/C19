import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf

def init():
	session = keras.backend.get_session()
	init = tf.global_variables_initializer()
	session.run(init)
	json_file = open('first.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load weights into new model
	loaded_model.load_weights("first.h5")
	print("Loaded Model from disk")

	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	graph = tf.compat.v1.get_default_graph()

	return loaded_model,graph,session
