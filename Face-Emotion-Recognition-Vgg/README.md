# Face-Emotion-Recognition

* This project is about Emotion Detection through facial expressions. CV can recognize and tell you what your emotion is by just looking at your facial expressions. It can detect whether you are angry, happy, sad, etc.

<img src="https://github.com/gayathri1462/Face-Emotion-Recognition/blob/main/output.png?raw=true.type" width="500" height="500">

* This computer vision model that we will build using Keras and VGG16 – a variant of Convolutional Neural Network. We will use this model to check the emotions in real-time using OpenCV and webcam. We will be working with Google Colab to build the model as it gives us the GPU and TPU. You can use any other IDE as well.

### The Dataset

* Dataset: https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge  

The name of the data set is fer2013 which is an open-source data set that was made publicly available for a Kaggle competition. It contains 48 X 48-pixel grayscale images of the face. There are seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral) present in the data. The CSV file contains two columns that are emotion that contains numeric code from 0-6 and a pixel column that includes a string surrounded in quotes for each image.

### Implementing VGG16 Network for Classification of Emotions with GPU
* First, we need to enable GPU in the Google Colab to get fast processing. We can enable it by going to ‘Runtime’ in Google Colab and then clicking on ‘Change runtime type’ and select GPU. Once it is enabled we will import the required libraries for building the network.
* Import all the libraries and now we will import the data set. I have already saved it in my drive so I will read it from there. 

### VGG16 Model for Emotion Detection
* Design the CNN model for emotion detection with different layers. We start with the initialization of the model followed by batch normalization layer and then different convents layers with ReLu as an activation function, max pool layers, and dropouts. 
* After this, we compile the model using Adam as an optimizer, loss as categorical cross-entropy, and metrics as accuracy.
* After compiling the model we then fit the data for training and validation. Here, we are taking the batch size to be 64 with 55 epochs. 
* Once the training has been done we can evaluate the model and compute loss and accuracy.
* We now serialize the model to JSON and save the model weights in an hd5 file so that we can make use of this file to make predictions rather than training the network again. 

### Testing the model in Real-time using OpenCV and Web Camera

* Test the model that we build for emotion detection in real-time using OpenCV and webcam. To do so we will write a python script. I used the Jupyter notebook in my local system to make use of a webcam. You can use other IDEs as well. 
* Install a few libraries that are required. After importing all the required libraries we will load the model weights that we saved earlier after training. load your saved model. After importing the model weights we have imported a haar cascade file that is designed by open cv to detect the frontal face.
* After importing the haar cascade file write a code to detect faces and classify the desired emotions. Assign the labels that will be different emotions like angry, happy, sad, surprise, neutral. As soon as you run the code,

* Using python real-time-detection.py

* A new window will pop up and your webcam will turn on. It will then detect the face of the person, draw a bounding box over the detected person, and then convert the RGB image into grayscale & classify it in real-time. 
* Please refer to the code for the same and sample outputs that are shown in the images. 
* To stop the running code press ‘q’. 




