import os
import numpy as np
import tensorflow as tf
from matplotlib import image
from PIL import Image
import pandas as pd

training_file = ''
testing_file = ''

train_dir = 'data/Train/'
test_dir = 'data/Test/'
test_csv = 'data/Test.csv'

def get_training_data():
    data = []
    classes = []
    for dirname in os.listdir(train_dir):
        if os.path.isdir(train_dir + dirname):
            image_class = int(dirname)
            dirname = dirname + '/'
            for filename in os.listdir(train_dir + dirname):
                if filename.endswith(".png"):
                    file_path = train_dir + dirname + filename
                    print("Processing ", filename, "with class", image_class)
                    img = Image.open(file_path)
                    img = img.resize((32,32))
                    data.append(np.asarray(img))
                    classes.append(image_class)
                    img.close()
    return (np.asarray(data, dtype=np.uint32) / 255, np.asarray(classes, dtype=np.uint32))

def get_testing_data():
    data = []
    df = pd.read_csv(test_csv)
    classes = np.asarray(df.ClassId, dtype=np.uint32)
    for filename in sorted(os.listdir(test_dir)):
                if filename.endswith(".png"):
                    file_path = test_dir + filename
                    img = Image.open(file_path)
                    img = img.resize((32,32))
                    data.append(np.asarray(img))
                    img.close()
    return (np.asarray(data, dtype=np.uint32) / 255, classes)