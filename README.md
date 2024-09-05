# PRODIGY_ML_03
Cat vs. Dog Image Classification using SVM
Overview
This project involves building a machine learning model to classify images of cats and dogs. Using Support Vector Machine (SVM) along with Principal Component Analysis (PCA) for dimensionality reduction, the model is trained on a dataset of images and evaluated for its performance in distinguishing between cats and dogs.

Dataset
The dataset used for this project is the https://www.kaggle.com/c/dogs-vs-cats/data. This dataset contains labeled images of cats and dogs, which are used for training and testing the model.

Project Details
Goals
To implement an image classification model using Support Vector Machines (SVM).
To utilize Principal Component Analysis (PCA) for dimensionality reduction.
To evaluate the model's performance with a confusion matrix and classification report.
Methodology
Data Extraction and Preprocessing:

Extracted the dataset from a ZIP file.
Loaded and resized images to a consistent size.
Normalized image pixel values and flattened images for model input.
Feature Extraction and Labeling:

Labels are assigned based on image filenames (cat or dog).
Images are transformed into feature vectors.


Model Training:

Implemented PCA for feature reduction.
Used Grid Search with Cross-Validation to find the best combination of PCA components and SVM kernel.
Evaluation:

Evaluated the model using accuracy, confusion matrix, and classification report.


Results:

The best model achieved an accuracy of approximately 67.62%.
Detailed evaluation metrics including precision, recall, and F1-score are provided in the classification report.
Code Overview
Data Preparation: data_preparation.py – Extracts, preprocesses, and labels images.
Model Training and Evaluation: model_training.py – Trains the model using Grid Search, evaluates it, and saves results.
Visualization: visualization.py – Generates and saves the confusion matrix.
Results
Best Parameters: {'pca__n_components': 0.9, 'svm__kernel': 'rbf'}
Best Score: 67.57%
Accuracy: 67.62%
Confusion Matrix and Classification Report are saved in the Dataset/ directory.


Here's a polished and attractive README file for your GitHub repository. It highlights the project's purpose, the dataset used, and the context of your internship. Feel free to customize it further to suit your preferences.

Cat vs. Dog Image Classification using SVM
Overview
This project involves building a machine learning model to classify images of cats and dogs. Using Support Vector Machine (SVM) along with Principal Component Analysis (PCA) for dimensionality reduction, the model is trained on a dataset of images and evaluated for its performance in distinguishing between cats and dogs.

Dataset
The dataset used for this project is the Dogs vs. Cats dataset from Kaggle. This dataset contains labeled images of cats and dogs, which are used for training and testing the model.

Project Details
Goals
To implement an image classification model using Support Vector Machines (SVM).
To utilize Principal Component Analysis (PCA) for dimensionality reduction.
To evaluate the model's performance with a confusion matrix and classification report.
Methodology
Data Extraction and Preprocessing:

Extracted the dataset from a ZIP file.
Loaded and resized images to a consistent size.
Normalized image pixel values and flattened images for model input.
Feature Extraction and Labeling:

Labels are assigned based on image filenames (cat or dog).
Images are transformed into feature vectors.
Model Training:

Implemented PCA for feature reduction.
Used Grid Search with Cross-Validation to find the best combination of PCA components and SVM kernel.
Evaluation:

Evaluated the model using accuracy, confusion matrix, and classification report.
Results:

The best model achieved an accuracy of approximately 67.62%.
Detailed evaluation metrics including precision, recall, and F1-score are provided in the classification report.
Code Overview
Data Preparation: data_preparation.py – Extracts, preprocesses, and labels images.
Model Training and Evaluation: model_training.py – Trains the model using Grid Search, evaluates it, and saves results.
Visualization: visualization.py – Generates and saves the confusion matrix.
Results
Best Parameters: {'pca__n_components': 0.9, 'svm__kernel': 'rbf'}
Best Score: 67.57%
Accuracy: 67.62%
Confusion Matrix and Classification Report are saved in the Dataset/ directory.
Example Outputs
Confusion Matrix:

![Confusion Matrix](Dataset/confusion matrix.png)

Classification Report:

yaml
Copy code
precision    recall  f1-score   support

       Cat       0.68      0.69      0.68      2529
       Dog       0.68      0.66      0.67      2471

    accuracy                           0.68      5000
   macro avg       0.68      0.68      0.68      5000
weighted avg       0.68      0.68      0.68      5000


Internship
This project is part of my internship at Prodigy Infotech. During this internship, I have gained valuable experience in applying machine learning techniques to real-world problems, including image classification.
