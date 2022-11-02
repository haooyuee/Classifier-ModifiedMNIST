# IFT6390_Kaggle_Competition_1 - ShengHaoyue
# Info
Name:               Haoyue Sheng 
Matricule:          20239178 
Kaggle username:    shenghaoyue

# Overview
- Generate classifier to classify minist modifier image. All images are spotted from two original mnist images, and the size of the image is 56 × 28 and the label is the sum of the two mnist values, i.e. 0 to 18. 
- The logreg method requires produce from scratch
- Other methods: decision tree; random forest; Support Vector Machines; ANN; CNN.
- Attachments include all method files(jupyter notebooks) and model files, except for the random forest its model file is too large.

# contents
├── Readme.md                   // help
├── classification-of-mnist-digits  // dataset from the kaggle <--need additional download
├── data_visualization.ipynb    
├── Logreg.ipynb                // logistic regression
|├── logreg0.1.json             // weight and bias for logreg
|├── logreg0.01.json            // weight and bias for logreg
|├── logreg3.json               // weight and bias for logreg
├── TreeForest.ipynb            // decision tree; random forest
|├── dtree.pickle               // model for decision tree
├── SVM.ipynb                   // Support Vector Machines
|├── SVM.pickle                 // model for SVM
├── ANN.ipynb                   // ANN
|├── ANN.h5                     // model for ANN
├── CNN.ipynb                   // CNN
|├── CNN.h5                     // model for CNN

# How to run
step1:
Download the dataset from the kaggle website and put the folder in the correct location as shown in the directory
    Classifying Handwritten Digits (Modified MNIST):
    https://www.kaggle.com/t/0d0b1c033ece47ffa1dbc8bd374689ae

step2:
Configure the right environment, for example: Colab (which I use)

step3:
To reproduce the project, run the following notebooks in the given order and follow the annotations:
    - `Logreg.ipynb`
    - `TreeForest.ipynb` 
    - `SVM.ipynb` 
    - `ANN.ipynb` 
    - `CNN.ipynb` 
