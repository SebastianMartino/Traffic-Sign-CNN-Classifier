import numpy as np
import tensorflow as tf
import os
import os.path
import pickle
from tensorflow.keras import Sequential
from preprocess import *
import datetime
import matplotlib.pyplot as plt


class CNN():
    def __init__(self):

        self.learning_rate = 1e-4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.hidden_dim1 = 1024
        self.hidden_dim2 = 512
        self.num_classes = 43
        self.num_batches = 100

        ## Followed Modified LeNet Architecture described in https://bayne.github.io/blog/post/project2-writeup/ with slight modification
        self.seq = Sequential(
            [
                ## 1st Convolution layer, 32 filters, kernel size of 1x1, relu activation
                tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu'),
                ## 2nd Convolution layer, 32 filters, kernel size of 5x5, relu activation
                tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu'),
                ## Max pooling, pool size of 2x2
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                ## 3rd Convolution layer, 32 filters, kernel size of 5x5, relu activation
                tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu'),
                ## Max pooling, pool size of 2x2
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                ## Flatten layer
                tf.keras.layers.Flatten(),
                ## 1st Fully connected layer, output size of hidden_dim1 (1024), relu activation
                tf.keras.layers.Dense(self.hidden_dim1, activation="relu"),
                ## Dropout layer, dropout rate of 0.6
                tf.keras.layers.Dropout(rate=0.6),
                ## 2nd Fully connected layer, output size of hidden_dim2 (512), relu activation
                tf.keras.layers.Dense(self.hidden_dim2, activation="relu"),
                ## Dropout layer, dropout rate of 0.6
                tf.keras.layers.Dropout(rate=0.6),
                ## 3rd and final Fully connected layer, output size of num_classes (43), softmax activation
                tf.keras.layers.Dense(self.num_classes, activation="softmax")
            ]
        )

def save_py_object(filename, object):
    output_file = open(filename, 'wb')
    pickle.dump(object, output_file, pickle.HIGHEST_PROTOCOL)
    output_file.close()

def main():
    ## Check if data has already previously been preprocessed and saved as pickle file
    if(os.path.exists('train_data.p') and os.path.exists('test_data.p')):
        print("Found saved preprocessed data")
        saved_train_data = open('train_data.p', 'rb')
        train_data = pickle.load(saved_train_data)
        saved_train_data.close()

        saved_test_data = open('test_data.p', 'rb')
        test_data = pickle.load(saved_test_data)
        saved_test_data.close()

    else: ## Otherwise, preprocess data and save it for future use
        print("No previously preprocessed data found")
        train_data = get_training_data()
        save_py_object('train_data.p', train_data)

        test_data = get_testing_data()
        save_py_object('test_data.p', train_data)
        
    model = CNN()
    
    ## Configure Model for training, specify optimizer, use sparse_categorical_crossentropy loss, evaluate accuracy metrics
    model.seq.compile(optimizer=model.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    ## Using Tensorboard for loss and accuracy visualization https://www.tensorflow.org/tensorboard/get_started
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    ## Shuffle inputs and labels
    indices = range(train_data[0].shape[0])
    shuffled_indices = tf.random.shuffle(indices)
    train_labels = tf.gather(train_data[1], shuffled_indices)
    train_inputs = tf.gather(train_data[0], shuffled_indices)
    
    ## Randomly flip training images along x axis
    train_inputs = tf.image.random_flip_left_right(train_inputs, train_inputs.shape[0])

    ## Train model for 10 epochs, attatch tensorboard_callback to get visualization of loss and accuracy during training
    model.seq.fit(train_inputs, train_labels, validation_data=(test_data[0], test_data[1]), epochs=10, batch_size=model.num_batches, callbacks=[tensorboard_callback])

    model.seq.summary()

if __name__ == '__main__':
    main()

