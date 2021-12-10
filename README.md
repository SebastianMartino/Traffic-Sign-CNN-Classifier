# Traffic-Sign-CNN-Classifier
Final Project for CSCI 1470 - DL nueral network to classify traffic signs

Results:
Training accuracy achieved after 10 epochs: ~87.5%
Validation accuracy achieved after 10 epochs: ~95%

The dataset used in training and testing this model, the GTSRB, can be found at https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Running the model:
python3 model.py

Expects dataset to be stored in folder named 'data' at the root directory of project.
If it is your first time running the model, the data will first have to be preprocessed which may take some time.
Preprocessed data is dumped as a pickle files 'train_data.p' and 'test_data.p' which will be utilized in future runs. 

Requires following non built-in libraries: Tensorflow, numpy, PIL Image, pandas, & matplotlib

Visualizing Results:
This project makes use of Tensorboard to log accuracy, loss and other information over the training process. After the model has been trained at least once, the command 'tensorboard --logdir logs/fit' can be used to create a local webserver at http://localhost:6006/ which displays visual information on past trainings.