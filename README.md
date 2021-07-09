# FaceMaskDetection
## Abstract
COVID-19 (Coronavirus) is an epidemic disease that has seriously affected the lives of all countries. The most important of the measures for the 
COVID-19 outbreak is to use a surgical face mask that covers the mouth and nose. In the study, images were defined and classified by working on 
deep learning techniques, and the detection was carried out by taking images of people with and without face masks.
## Introduction
The main aim of this study is by preprocessing on images taking from dataset, of face with mask and without mask person classification and detection to provide by deep learning method. the spread of the epidemic This model, designed to prevent detection of people via real-time video stream is doing. This project uses OpenCV TensorFlow libraries It is written using the Python programming language.The masked and unmasked two-category dataset used model with an additional dataset divided into three categories.detection is also considered. So, data to be used in mask detection is correctly masked face, false masked face, and unmasked face dataset classified as.
There are two goals of using these datasets:
i) To identify people wearing and not wearing masks,
ii) Whether the people wearing the mask are wearing the mask correctly determine.
Face mask detection since the beginning of the epidemic many studies have been developed on most distinguish whether there is a mask or not. Given the current situation and new variants, we aim to achieve a more sensitive detection model to strengthen mask wearing practice worldwide.
## Dataset
The Kaggle dataset, which is considered suitable for model training, consists of a total of 7553 images, 3725 using masks and 3828 not using masks. Images containing multiple faces, masks of different types and different colors are in the dataset. Face images taken from different perspectives are also included in the dataset. Therefore, it is a dataset that is thought to be very useful and productive for training.

<img src="https://user-images.githubusercontent.com/60175396/125008103-72ac9400-e06a-11eb-9eec-bafae68864be.png"><br>
<sub><b>Figure 1: </b> Kaggle Face Mask Dataset </sub><br>

The other dataset used in the study is the MaskedFaceNet dataset, which contains CMFD (Correctly Masked Face Dataset), IMFD (Incorrectly Masked Face Dataset) and RMFD (Real Masked Face Dataset). There are 962 correctly masked, 955 incorrectly masked and 953 unmasked face images in the dataset containing 2870 images in total. IMFD and CMFD images are synthetically generated. The dataset contains different human faces in terms of ethnicity, age, gender. At the same time, it has diversity in the outdoor environment such as the lighting background. Images are of high quality and 1024x1024 resolution. The main reason for choosing the dataset was that the model was requested to be detailed with the help of various tags. The two classes (masked and unmasked) observed in other datasets are divided into 3 classes in this dataset.

<img src="https://user-images.githubusercontent.com/60175396/125008373-10a05e80-e06b-11eb-835e-81249e85a256.png"><br>
<sub><b>Figure 2: </b> RMFD,CMFD,IMFD dataset </sub><br>

## Preprocess
By preprocessing, the data is made available for further use. The success of this phase allows the model to be successfully trained and the result to be more accurate. Data preprocessing involves resizing the image, converting the image to an array, and converting the labeled data into a numeric label. Resizing images is a critical preprocessing step for the effectiveness of training models and machine learning. The smaller the size of the image, the better the model works. In this project, the resizing process is to resize all of the images to 224 × 224 pixels.
## Data Visualization
All the images in the Maskedface-Net dataset are divided into three categories as correctly masked, incorrectly masked and unmasked, and the images in the Kaggle dataset are divided into two categories as masked and unmasked.


<img src="https://user-images.githubusercontent.com/60175396/125008636-91f7f100-e06b-11eb-8a64-efa0e71d96e4.png"><br>
<sub><b>Figure 3: </b> Masked Face-Net dataset distribution </sub><br>

<img src="https://user-images.githubusercontent.com/60175396/125008788-ec914d00-e06b-11eb-9dd9-86aac63ff182.png"><br>
<sub><b>Figure 4: </b> Kaggle Face Mask dataset distribution </sub><br>

<img src="https://user-images.githubusercontent.com/60175396/125008868-1480b080-e06c-11eb-9668-c71a04dcbae6.png"><br>
<sub><b>Figure 5: </b> Aspect ratio histogram </sub><br>

The aspect ratio histogram of the images in the dataset is shown in Figure 5. Due to the disproportionate size of the images, resizing was done with data preprocessing.

## Training
In the project, face mask recognition was performed with the ResNet50 image classification method on the binary classified data set. This model is an enhanced version of convolutional neural networks (CNN). Resnet50 is a 50-layer network trained on the ImageNet dataset. In the Resnet model, bottleneck blocks are used to train faster.

Face mask recognition was performed with MobilNetV2 method on the multi-classified dataset. MobileNetV2 is a method based on Convolutional Neural Network (CNN) developed by Google with performance and optimization to be more efficient.

After our model is fully trained with the provided dataset, it is tested to see the prediction result of the model. As a result of the training, the loss value decreased regularly in both models, and the accuracy value increased to the desired level. There was no deviation in val_loss and val_acc values, indicating that overfitting did not occur. With the early stop, the training has ended successfully. Since the straight line was stable, no more epochs were needed for improvement.

## Experiments and Discussion
Another model applied on this dataset is the ResNet model, which is a transfer learning method. When using the transfer learning technique, only the last layer of the previously trained model is modified to suit the new model to be trained. This ensures high success. At the same time, the success performance of the model has been increased by preprocessing on the data. The effect of these processes on the result can be seen in Figure 9.

<img src="https://user-images.githubusercontent.com/60175396/125008972-4e51b700-e06c-11eb-897b-ce350abb1096.png"><br>
<sub><b>Figure 6: </b> ResNet model acc-loss graph </sub><br>

Confusion matrix is a technique used to measure the success of the classification model. For binary classification, there are four values in the confision matrix. The accuracy value calculated using these values was 0.99, and the error rate was 0.003.

<img src="https://user-images.githubusercontent.com/60175396/125009057-7b9e6500-e06c-11eb-8b8b-58e542da7869.png"><br>
<sub><b>Figure 7: </b> ResNet confusion matris </sub><br>

Another technique used to measure performance for classification models is the AUC-ROC curve. The closer the ROC curve is to the TP axis, the higher the accuracy. Since the accuracy rate of the model is 0.99, it can be seen in Figure 8 that the curve obtained is very close to the upper left corner.

<img src="https://user-images.githubusercontent.com/60175396/125009169-a983a980-e06c-11eb-8b31-a0c3c8e2df0e.png"><br>
<sub><b>Figure 8: </b> ResNet AUC-ROC curve </sub><br>

A MobileNet model was created for the three-label dataset. Training was conducted using this dataset to create a sensitive detection model.

<img src="https://user-images.githubusercontent.com/60175396/125009305-f798ad00-e06c-11eb-87d2-f4522b31ee7d.png"><br>
<sub><b>Figure 9: </b> MobileNet acc-loss graph </sub><br>

As seen in Picture 9, there is no gap between the train and validation values. This indicates that there is no overfitting in the model. Successful acc-loss values were obtained in this training.

In the real-time test of the model, it has been determined that it has difficulty in detecting in low-light environments. For this reason, another training was carried out by applying a lighting filter to the data. After blackening the images, both the success of the model increased and better results were obtained in real-time tests.


<img src="https://user-images.githubusercontent.com/60175396/125009414-3169b380-e06d-11eb-9c8f-be3b4e8b0e68.png"><br>
<sub><b>Figure 10: </b> MobileNet confusion matris </sub><br>

There are nine values in the confusion matrix for the triple classification model. With the help of these values, the accuracy value was calculated as 0.98 and the error rate as 0.013.

## Result
With deep learning, mask-wearing and non-mask-wearing people have been successfully detected in face mask detection. A more sensitive
In order to perform the detection, a dataset divided into three classes was added to the study. Datasets containing binary labels (masked, unmasked) and triple labels (masked, unmasked and false masked) were used, which were deemed appropriate for the project. CNN and ResNet models were created for the two-class dataset, and the MobileNet model for the three-class dataset. Here, the success rate with ResNet is 99% and the success rate with MobileNet is 96%. This performance has also been successfully observed in the real-time test of the model.
