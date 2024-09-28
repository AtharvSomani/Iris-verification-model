# -*- coding: utf-8 -*-
"""LEFTandRIGHTirisModel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19cf4mQ2_04DFDflaWlc3W-kdgP1wLeqG
"""

import tensorflow as tf
import tensorflow as tf
import os
import glob
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Layer, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import Loss

import tensorflow as tf
import os
import glob
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Layer, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import Loss


class AbsoluteDifferenceLayer(Layer):
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize to (128, 128)
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

def combine_iris_images(data_root):
    """
    Combines left and right iris images for each subject into a list.

    Args:
        data_root: Path to the root directory of the dataset.

    Returns:
        A list of tuples, where each tuple contains:
            - subject_id
            - list of image paths (combined left and right)
    """

    subject_data = []
    for subject_dir in os.listdir(data_root):
        subject_path = os.path.join(data_root, subject_dir)
        if os.path.isdir(subject_path):
            subject_id = subject_dir
            image_paths = sorted(glob.glob(os.path.join(subject_path, "*_L.*"))) + sorted(glob.glob(os.path.join(subject_path, "*_R.*")))
            subject_data.append((subject_id, image_paths))
    return subject_data



def create_cnn():
    inputs = Input(shape=(64, 64, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs, x)



data_root = "/content/drive/MyDrive/train"  # Replace with your actual data root
combined_data = combine_iris_images(data_root)

from sklearn.model_selection import train_test_split
# Define a new model that takes two inputs (left and right iris images)
def create_verification_model():
    cnn = create_cnn()
    input_left = Input(shape=(64, 64, 3))
    input_right = Input(shape=(64, 64, 3))
    feat_left = cnn(input_left)
    feat_right = cnn(input_right)
    distance = AbsoluteDifferenceLayer()([feat_left, feat_right])
    output = Dense(1, activation='sigmoid')(distance)
    return Model(inputs=[input_left, input_right], outputs=output)

# Create the verification model
verification_model = create_verification_model()

# Compile the model with binary cross-entropy loss and RMSprop optimizer
verification_model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
# Define a function to create pairs of left and right iris images with corresponding labels
def create_pairs(data):
    pairs = []
    labels = []
    for i, (subject_id, images) in enumerate(data):
        for j in range(len(images)):
            left_image = load_and_preprocess_image(images[j])
            for k in range(j + 1, len(images)):
                right_image = load_and_preprocess_image(images[k])
                pairs.append([left_image, right_image])
                labels.append(1)  # Same person
            for k in range(i + 1, len(data)):
                negative_subject_id, negative_images = data[k]
                negative_index = np.random.randint(len(negative_images))
                right_image = load_and_preprocess_image(negative_images[negative_index])
                pairs.append([left_image, right_image])
                labels.append(0)  # Different person
    return np.array(pairs), np.array(labels)

# Create pairs of left and right iris images with corresponding labels
pairs, labels = create_pairs(combined_data)

# Split the pairs and labels into training and testing sets
train_pairs, test_pairs, train_labels, test_labels = train_test_split(pairs, labels, test_size=0.2, random_state=42)

# Train the verification model
verification_model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels, epochs=5, batch_size=32)

# Evaluate the verification model
loss_and_metrics = verification_model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels, verbose=0)
loss = loss_and_metrics[0]
accuracy = loss_and_metrics[1]
print('Loss:', loss)
print('Accuracy:', accuracy)

# Define a function to verify whether two iris images are of the same person
def verify_iris_images(left_image_path, right_image_path):
    left_image = load_and_preprocess_image(left_image_path)
    right_image = load_and_preprocess_image(right_image_path)
    left_image = left_image.reshape((1, 64, 64, 3))
    right_image = right_image.reshape((1, 64, 64, 3))
    prediction = verification_model.predict([left_image, right_image])
    if prediction > 0.5:
        print('The two iris images are likely of the same person!')
    else:
        print('The two iris images are likely of different people!')

# Test the verification model with some example images
left_image_path = '/content/drive/MyDrive/train/014/01_L.bmp'
right_image_path = '/content/drive/MyDrive/train/017/09_R.bmp'
verify_iris_images(left_image_path, right_image_path)

import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr  # FNR is 1 - TPR
    eer_threshold = thresholds[np.nanargmin(np.absolute((fpr - fnr)))]
    eer = fpr[np.nanargmin(np.absolute((fpr - fnr)))]
    return eer, eer_threshold, fpr, tpr

# Step 1: Get the predicted probabilities for the test pairs
y_scores = verification_model.predict([test_pairs[:, 0], test_pairs[:, 1]])

# Step 2: Calculate the EER
eer, eer_threshold, fpr, tpr = calculate_eer(test_labels, y_scores)

# Step 3: Output the EER and the threshold at which it occurs
print(f"Equal Error Rate (EER): {eer}")
print(f"Threshold at EER: {eer_threshold}")

# Step 4: Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {np.trapz(tpr, fpr):.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (chance level)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)

# Highlight the EER point on the ROC curve
eer_fpr = fpr[np.nanargmin(np.absolute((fpr - (1 - tpr))))]
eer_tpr = tpr[np.nanargmin(np.absolute((fpr - (1 - tpr))))]
plt.plot(eer_fpr, eer_tpr, marker='o', markersize=8, color='red', label=f'EER = {eer:.2f}')
plt.legend()

# Show the plot
plt.show()

verification_model.save('iris_verification_model.h5', save_format='h5')

left_image_path = '/content/drive/MyDrive/train/014/01_L.bmp'
right_image_path = '/content/drive/MyDrive/train/017/09_R.bmp'
verify_iris_images(left_image_path, right_image_path)

# Define a function to automate testing of verify_iris_images
def automate_testing(image_dir):
    # Iterate through all possible pairs of left and right iris images
    for subject_id in os.listdir(image_dir):
        subject_dir = os.path.join(image_dir, subject_id)
        if os.path.isdir(subject_dir):
            left_images = [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) if f.endswith("_L.bmp")]
            right_images = [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) if f.endswith("_R.bmp")]

            for left_image_path in left_images:
                for right_image_path in right_images:
                    print(f"Testing: {left_image_path} and {right_image_path}")
                    verify_iris_images(left_image_path, right_image_path)
                    print()

# Example usage: Automate testing with all images in a directory
image_dir = '/content/drive/MyDrive/train'  # Change this to your directory

automate_testing(image_dir)

import os

def automate_inter_subject_testing(image_dir):
    """
    Automates inter-subject testing by comparing iris images from different subjects.

    Args:
        image_dir: The directory containing subject directories with iris images.
    """

    subject_dirs = [os.path.join(image_dir, subject_id) for subject_id in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, subject_id))]

    for subject_dir1 in subject_dirs:
        left_images1 = [os.path.join(subject_dir1, f) for f in os.listdir(subject_dir1) if f.endswith('_L.bmp')]
        for subject_dir2 in subject_dirs:
            if subject_dir1 != subject_dir2:  # Avoid comparing images within the same subject
                right_images2 = [os.path.join(subject_dir2, f) for f in os.listdir(subject_dir2) if f.endswith('_R.bmp')]
                for left_image_path in left_images1:
                    for right_image_path in right_images2:
                        print(f"Testing: {left_image_path} and {right_image_path}")
                        verify_iris_images(left_image_path, right_image_path)
                        print()

# Example usage
image_dir = '/content/drive/MyDrive/train'
automate_inter_subject_testing(image_dir)

from PIL import Image

def verify_iris_images(left_image_path, right_image_path):
    left_image = load_and_preprocess_image(left_image_path)
    right_image = load_and_preprocess_image(right_image_path)
    left_image = left_image.reshape((1, 64, 64, 3))
    right_image = right_image.reshape((1, 64, 64, 3))
    prediction = verification_model.predict([left_image, right_image])
    if prediction > 0.5:
        verification_result = "Same person"
    else:
        verification_result = "Different people"
    return verification_result, prediction

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize to (128, 128)
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

pip install openpyxl

pip install onnx

pip install keras2onnx

pip install tf2onnx

import tensorflow as tf

# Define a custom AbsoluteDifferenceLayer
class AbsoluteDifferenceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AbsoluteDifferenceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

    def get_config(self):
        config = super(AbsoluteDifferenceLayer, self).get_config()
        return config

# Load the Keras model from the .h5 file
model = tf.keras.models.load_model(
    '/content/iris_verification_model.h5',
    custom_objects={'AbsoluteDifferenceLayer': AbsoluteDifferenceLayer}
)

# Convert the Keras model to an ONNX model
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name='input')])

# Save the ONNX model to a file
onnx.save(onnx_model, 'model.onnx')

import numpy as np

import os
import pandas as pd

def verify_iris_images(left_image_path, right_image_path):
    """
    Function to verify iris images by comparing them.
    Args:
        left_image_path (str): Path to the left iris image.
        right_image_path (str): Path to the right iris image.
    Returns:
        tuple: Verification result ("Same person" or "Different people") and prediction score.
    """
    # Load and preprocess the images (replace with actual implementation)
    left_image = load_and_preprocess_image(left_image_path)
    right_image = load_and_preprocess_image(right_image_path)

    # Reshape images to match model input requirements
    left_image = left_image.reshape((1, 64, 64, 3))
    right_image = right_image.reshape((1, 64, 64, 3))

    # Predict using the verification model
    prediction = verification_model.predict([left_image, right_image])

    # Determine if the images are of the same person or different people
    if prediction > 0.5:
        verification_result = "Same person"
    else:
        verification_result = "Different people"

    return verification_result, prediction

def automate_testing(image_dir):
    """
    Automates intra-subject and inter-subject testing by comparing iris images
    and stores the results in an Excel file.

    Args:
        image_dir: The directory containing subject directories with iris images.
    """
    results = []

    # List of subject directories
    subject_dirs = [os.path.join(image_dir, subject_id) for subject_id in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, subject_id))]

    # Intra-subject testing (same subject)
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        left_images = [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) if f.endswith('_L.bmp')]
        right_images = [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) if f.endswith('_R.bmp')]

        for left_image_path in left_images:
            for right_image_path in right_images:
                verification_result, prediction = verify_iris_images(left_image_path, right_image_path)
                print(f"Intra-subject Test | Subject: {subject_id} | Left: {left_image_path} | Right: {right_image_path} | Result: {verification_result} | Score: {prediction[0][0]}")
                results.append({'Subject 1': subject_id, 'Subject 2': subject_id, 'Left Image': left_image_path, 'Right Image': right_image_path, 'Verification Result': verification_result, 'Prediction Score': prediction, 'Type': 'Intra-subject'})

    # Inter-subject testing (different subjects)
    for subject_dir1 in subject_dirs:
        subject_id1 = os.path.basename(subject_dir1)
        left_images1 = [os.path.join(subject_dir1, f) for f in os.listdir(subject_dir1) if f.endswith('_L.bmp')]
        for subject_dir2 in subject_dirs:
            if subject_dir1 != subject_dir2:
                subject_id2 = os.path.basename(subject_dir2)
                right_images2 = [os.path.join(subject_dir2, f) for f in os.listdir(subject_dir2) if f.endswith('_R.bmp')]
                for left_image_path in left_images1:
                    for right_image_path in right_images2:
                        verification_result, prediction = verify_iris_images(left_image_path, right_image_path)
                        print(f"Inter-subject Test | Subject 1: {subject_id1} | Subject 2: {subject_id2} | Left: {left_image_path} | Right: {right_image_path} | Result: {verification_result} | Score: {prediction[0][0]}")
                        results.append({'Subject 1': subject_id1, 'Subject 2': subject_id2, 'Left Image': left_image_path, 'Right Image': right_image_path, 'Verification Result': verification_result, 'Prediction Score': prediction, 'Type': 'Inter-subject'})

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save the results to an Excel file
    output_file = os.path.join(image_dir, 'iris_testing_results.xlsx')
    df.to_excel(output_file, index=False)
    print(f'Results saved to {output_file}')

# Example usage
image_dir = '/content/drive/MyDrive/train'  # Change this to your directory
automate_testing(image_dir)