# testing accuracy on test dataset
from sklearn.metrics import accuracy_score
from keras.models import load_model
import pandas as pd
from PIL import Image 
import numpy as np

model = load_model('traffic_classifier_4.h5')
y_test = pd.read_csv('test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]

for img in imgs:
    image = Image.open(img).convert('RGB')
    image = image.resize((30,30))
    data.append(np.array(image))

X_test=np.array(data)

pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)

#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))
