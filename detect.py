import cv2
import numpy as np
import copy
import math
import datetime

from keras.preprocessing import image as image_utils

def predict_rgb_image(img):
	result = gesture_names[model.predict_classes(img)[0]]
	print(result)
	return (result)

from keras.models import load_model
model = load_model('/Users/brenner/project_kojak/drawing_VGG.h5')

gesture_names = {0: 'C',
				 1: 'Fist',
				 2: 'L',
				 3: 'Okay',
				 4: 'Palm',
				 5: 'Peace'}

def predict_rgb_image_vgg(img):
	# img2rgb = image_utils.load_img(path=path, target_size=(224, 224))
	# img2rgb = image_utils.img_to_array(img2rgb)
#     image_rgb.append(img2rgb)
#     img2rgb = img2rgb.reshape(1, 224, 224, 3)
	img = np.array(img, dtype='float32')
	img /= 255
	pred_array = model.predict(img)
	print(f'pred_array: {pred_array}')
	result = gesture_names[np.argmax(pred_array)]
	print(f'Result: {result}')
	print(max(pred_array[0]))
	print(result)
	return result