# Iris-verification-model

# Left and Right Iris Verification Model
This project implements a Convolutional Neural Network (CNN) using TensorFlow to verify whether the left and right iris images belong to the same person. The model processes paired images, learns their features, and predicts whether both irises are from the same individual.

# Features
Model Architecture: A custom CNN built from scratch using TensorFlow. The model takes in pairs of iris images (left and right) and outputs a similarity score, indicating whether both irises belong to the same person.
Image Processing: Input images are processed using PIL, resized, and preprocessed to be used in the model.
Verification: The model is trained to differentiate between irises of the same individual versus those of different individuals.
