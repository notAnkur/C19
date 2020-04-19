import numpy as np #manupulate arrays
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator #tries to generate multiple images out of 1
from keras.models import Sequential #linear stack of layers
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import backend as K
#activationfn(when to activate to 1,-1,0),dropout-helps nn not overfit,pooling-reduce size of data
#flatten- 2d image to 1 image, dense- hidden layer,backend- which layer is coming first?
img_width,img_height=150,150
train_data='DATA/Train'
validation_data='DATA/Validation'
train_samples=150
validation_samples=50
epochs=60
batch_size=5

if K.image_data_format()=='channels_first':
	input_shape=(3,img_width,img_height)
else:
	input_shape=(img_width,img_height,3)

#Generating many images out of 1 for train images
train_increase=ImageDataGenerator(
	shear_range=0.2,
	zoom_range=0.1,
	vertical_flip=True,
	horizontal_flip=True,
	rescale=1./255
	)


test_increase=ImageDataGenerator(rescale=1./255)

train_generator=train_increase.flow_from_directory(
	train_data,
	target_size=(img_width,img_height),
	batch_size=batch_size,
	class_mode='binary')

validation_generator=test_increase.flow_from_directory(
	validation_data,
	target_size=(img_width,img_height),
	batch_size=batch_size,
	class_mode='binary')

detector=Sequential()
detector.add(Conv2D(32,(3,3),input_shape=input_shape))
detector.add(Activation('relu'))
detector.add(MaxPooling2D(pool_size=(2,2)))

detector.summary()

detector.add(Conv2D(32,(3,3)))
detector.add(Activation('relu'))
detector.add(MaxPooling2D(pool_size=(2,2)))

detector.add(Conv2D(64,(3,3)))
detector.add(Activation('relu'))
detector.add(MaxPooling2D(pool_size=(2,2)))

detector.add(Flatten())
detector.add(Dense(64))
detector.add(Activation('relu'))
detector.add(Dropout(0.5))
detector.add(Dense(1))
detector.add(Activation('sigmoid'))

detector.summary()

detector.compile(loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])

detector.fit_generator(
	train_generator,
	steps_per_epoch=train_samples//batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=validation_samples//batch_size)

detector.save_weights('first.h5')
# serializing as json
# blah blah json blah blah
detector_json = detector.to_json()
with open("first.json", "w") as json_file:
    json_file.write(detector_json)

img_pred=image.load_img('DATA/Validation/NONCovid/5.jpeg',target_size=(150,150))
img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred,axis=0)

result=detector.predict(img_pred)
print(result)
if result[0][0]==1:
	prediction="NONCOVID"
else:
	prediction="COVID"

print(prediction)
