#coding=utf-8
import IPython
import json, os, re, sys, time
import numpy as np
import imageio
import scipy.misc
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import keras.backend as K
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
K.set_session(session)


def predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

if __name__ == '__main__':
    model_path = sys.argv[1]
    print('Loading model:', model_path)
    t0 = time.time()
    model = load_model(model_path)
    t1 = time.time()
    print('Loaded in:', t1-t0)

    test_path = sys.argv[2]
    print('Generating predictions on image:', sys.argv[2])
    my_image = imageio.imread(sys.argv[2])
    imshow(my_image)
    preds = predict(sys.argv[2], model)
    print(preds)
    if(preds[0][0] > preds[0][1]):
        print ("猫")
    else:
        print("狗")
    plt.show()

