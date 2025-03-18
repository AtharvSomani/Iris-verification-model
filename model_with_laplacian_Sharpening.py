import tensorflow as tf
import os
import glob
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Layer, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from tensorflow.keras.utils import img_to_array


# Custom layer for absolute difference
class AbsoluteDifferenceLayer(Layer):
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

def load_and_preprocess_image(image_path):
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize to match model input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Convert to grayscale and apply Laplacian for sharpening
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)  # Compute Laplacian
    laplacian = np.uint8(np.absolute(laplacian))  # Convert back to uint8
    
    # Convert Laplacian to 3 channels and blend with the original image
    laplacian_colored = np.stack([laplacian] * 3, axis=-1)  
    sharpened = cv2.addWeighted(image, 1.5, laplacian_colored, -0.5, 0)  # Adjust weights as needed

    # Convert to array format and normalize
    final_image = img_to_array(sharpened) / 255.0  

    return final_image


# Function to combine iris images from data_root
def combine_iris_images(data_root):
    subject_data = []
    for subject_dir in os.listdir(data_root):
        subject_path = os.path.join(data_root, subject_dir)
        if os.path.isdir(subject_path):
            subject_id = subject_dir
            # Look for both left and right eye images
            image_paths = sorted(glob.glob(os.path.join(subject_path, "*_L.*"))) + \
                          sorted(glob.glob(os.path.join(subject_path, "*_R.*")))
            subject_data.append((subject_id, image_paths))
    return subject_data

# Define the CNN architecture
def create_cnn():
    inputs = Input(shape=(64, 64, 3))
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    return Model(inputs, x)

# Set the path to your dataset
data_root = r"C:\Atharv/MTech_Project/train"  # Use raw-string if using backslashes in Windows paths
combined_data = combine_iris_images(data_root)

# Split data into train, validation, and test sets
train_data, temp_data = train_test_split(combined_data, test_size=0.4, random_state=42)  # 60% train, 40% temp
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 20% val, 20% test

# Create pairs of images and labels for training, validation, and testing
def create_pairs(data):
    pairs = []
    labels = []
    for i, (subject_id, images) in enumerate(data):
        # Create intra-class pairs (same person)
        for j in range(len(images)):
            left_image = load_and_preprocess_image(images[j])
            for k in range(j + 1, len(images)):
                right_image = load_and_preprocess_image(images[k])
                pairs.append([left_image, right_image])
                labels.append(1)  # Same person
            # Create inter-class pairs (different persons)
            for k in range(i + 1, len(data)):
                negative_subject_id, negative_images = data[k]
                negative_index = np.random.randint(len(negative_images))
                right_image = load_and_preprocess_image(negative_images[negative_index])
                pairs.append([left_image, right_image])
                labels.append(0)  # Different person
    return np.array(pairs), np.array(labels)

# Create pairs for train, validation, and test sets
train_pairs, train_labels = create_pairs(train_data)
val_pairs, val_labels = create_pairs(val_data)
test_pairs, test_labels = create_pairs(test_data)

# Define the verification model using the Siamese network structure
def create_verification_model():
    cnn = create_cnn()
    input_left = Input(shape=(64, 64, 3))
    input_right = Input(shape=(64, 64, 3))
    feat_left = cnn(input_left)
    feat_right = cnn(input_right)
    distance = AbsoluteDifferenceLayer()([feat_left, feat_right])
    output = Dense(1, activation='sigmoid')(distance)
    return Model(inputs=[input_left, input_right], outputs=output)

verification_model = create_verification_model()

# Define callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Compile the model
verification_model.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(learning_rate=0.001), 
    metrics=['accuracy']
)

# Train the model
verification_model.fit(
    [train_pairs[:, 0], train_pairs[:, 1]],
    train_labels,
    epochs=100,
    batch_size=32,
    validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on the test set
loss_and_metrics = verification_model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels, verbose=0)
loss = loss_and_metrics[0]
accuracy = loss_and_metrics[1]
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Function to verify whether two iris images are of the same person
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

# Example verification test
left_image_path = r"C:\Atharv/MTech_Project/train/015/02_L.bmp"
right_image_path = r"C:\Atharv/MTech_Project/train/038/06_R.bmp"
verify_iris_images(left_image_path, right_image_path)

# Save the trained model
verification_model.save('iris_verification_model_With_Laplacian_Sharpening.h5', save_format='h5')

# Calculate Equal Error Rate (EER) and plot ROC curve
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr  # FNR is 1 - TPR
    eer_threshold = thresholds[np.nanargmin(np.absolute((fpr - fnr)))]
    eer = fpr[np.nanargmin(np.absolute((fpr - fnr)))]
    return eer, eer_threshold, fpr, tpr

# Predicted probabilities for the test pairs
y_scores = verification_model.predict([test_pairs[:, 0], test_pairs[:, 1]])

# Calculate the EER
eer, eer_threshold, fpr, tpr = calculate_eer(test_labels, y_scores)
print(f"Equal Error Rate (EER): {eer}")
print(f"Threshold at EER: {eer_threshold}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {np.trapz(tpr, fpr):.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)

# Highlight the EER point on the ROC curve
eer_index = np.nanargmin(np.absolute((fpr - (1 - tpr))))
eer_fpr = fpr[eer_index]
eer_tpr = tpr[eer_index]
plt.plot(eer_fpr, eer_tpr, marker='o', markersize=8, color='red', label=f'EER = {eer:.2f}')
plt.legend()
plt.show()

# Functions for generating and saving saliency maps for model interpretability
def generate_saliency_map(model, left_image, right_image):
    """Generate saliency maps for the input images."""
    left_image = tf.expand_dims(left_image, axis=0)
    right_image = tf.expand_dims(right_image, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(left_image)
        tape.watch(right_image)
        prediction = model([left_image, right_image])

    grads_left = tape.gradient(prediction, left_image)
    grads_right = tape.gradient(prediction, right_image)

    saliency_left = tf.reduce_max(tf.abs(grads_left), axis=-1).numpy()[0]
    saliency_right = tf.reduce_max(tf.abs(grads_right), axis=-1).numpy()[0]

    return saliency_left, saliency_right

def save_saliency_maps(data, model, output_dir="saliency_maps", num_folders=5):
    """Generate and save saliency maps for the first few folders."""
    os.makedirs(output_dir, exist_ok=True)
    intra_dir = os.path.join(output_dir, "intra_class")
    inter_dir = os.path.join(output_dir, "inter_class")
    os.makedirs(intra_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)

    folder_count = 0
    for subject_id, images in data:
        if folder_count >= num_folders:
            break
        folder_count += 1
        print(f"Processing folder: {subject_id}")

        # Intra-class testing: compare different images of the same subject
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                left_image = load_and_preprocess_image(images[i])
                right_image = load_and_preprocess_image(images[j])

                saliency_left, saliency_right = generate_saliency_map(model, left_image, right_image)

                fig, axes = plt.subplots(2, 3, figsize=(12, 6))
                axes[0, 0].imshow(left_image)
                axes[0, 0].set_title("Left Image")
                axes[0, 1].imshow(right_image)
                axes[0, 1].set_title("Right Image")
                axes[0, 2].axis("off")

                axes[1, 0].imshow(saliency_left, cmap='hot')
                axes[1, 0].set_title("Left Saliency Map")
                axes[1, 1].imshow(saliency_right, cmap='hot')
                axes[1, 1].set_title("Right Saliency Map")
                axes[1, 2].axis("off")

                plt.tight_layout()
                file_name = f"intra_{subject_id}_{i}_{j}.png"
                plt.savefig(os.path.join(intra_dir, file_name))
                plt.close()

        # Inter-class testing: compare one image from this subject to one from another subject
        for other_subject_id, other_images in data:
            if subject_id == other_subject_id:
                continue

            left_image = load_and_preprocess_image(images[0])
            right_image = load_and_preprocess_image(other_images[0])

            saliency_left, saliency_right = generate_saliency_map(model, left_image, right_image)

            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            axes[0, 0].imshow(left_image)
            axes[0, 0].set_title("Left Image")
            axes[0, 1].imshow(right_image)
            axes[0, 1].set_title("Right Image")
            axes[0, 2].axis("off")

            axes[1, 0].imshow(saliency_left, cmap='hot')
            axes[1, 0].set_title("Left Saliency Map")
            axes[1, 1].imshow(saliency_right, cmap='hot')
            axes[1, 1].set_title("Right Saliency Map")
            axes[1, 2].axis("off")

            plt.tight_layout()
            file_name = f"inter_{subject_id}_vs_{other_subject_id}.png"
            plt.savefig(os.path.join(inter_dir, file_name))
            plt.close()
            break  # Limit to one inter-class comparison per folder

# Generate and save saliency maps for a few folders in the test set
save_saliency_maps(test_data[:5], verification_model)
