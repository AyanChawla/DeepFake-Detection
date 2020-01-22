import cv2
import  pandas as pd
from keras.models import load_model
import numpy as np
imagePath = input("enter the path of the image file: ")
image=cv2.imread(imagePath)
img=np.array(image)
model = load_model('model.h5')
pred = model.predict(img)
y=pd.DataFrame(y)
y=y.to_json(r'final.json')

''' To check the image uncomment this part of code

cv2.imshow('Test Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
