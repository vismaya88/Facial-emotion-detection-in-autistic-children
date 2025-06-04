import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

emotion_labels = ['anger', 'fear', 'joy', 'Natural', 'sadness', 'surprise']
num_classes = len(emotion_labels)

def load_data(data_dir):
    images = []
    labels = []

    for label in emotion_labels:
        label_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (96, 96))  
                images.append(image)
                labels.append(label)

    images = np.array(images, dtype='float32') / 255.0  
    labels = np.array(labels)

    lb = LabelBinarizer()  
    labels = lb.fit_transform(labels)

    return images, labels, lb
train_data_dir = 'dataset/train'


def build_model(input_shape, num_classes, model_type='MobileNetV2'):
    if model_type == 'MobileNetV2':
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif model_type == 'VGG16':
        base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    elif model_type == 'ResNet50':
        base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    
    base_model.trainable = False  

    # Add custom layers on top of the base model
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

input_shape = (96, 96, 3)
train_data_dir = 'dataset/train'
train_images, train_labels, lb = load_data(train_data_dir)

# Calculate class weights to handle imbalance
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(train_labels, axis=1)),
    y=np.argmax(train_labels, axis=1)
)

class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
models_to_compare = ['MobileNetV2', 'VGG16', 'ResNet50']
results = {}

# evaluation of each models
for model_type in models_to_compare:
    model = build_model(input_shape, num_classes, model_type)
    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Training {model_type}...")
    history = model.fit(
        train_images, train_labels,
        epochs=3,  
        batch_size=16,  
        validation_split=0.2,
        class_weight=class_weights
    )

    # Saving the trained model
    model.save(f"{model_type}_model.h5")
    print(f"{model_type} model saved as {model_type}_model.h5")

    # Saving the results
    train_predictions = model.predict(train_images)
    train_predictions = np.argmax(train_predictions, axis=1)
    train_labels_argmax = np.argmax(train_labels, axis=1)

    # Classification report and confusion matrix
    report = classification_report(train_labels_argmax, train_predictions, target_names=emotion_labels, output_dict=True)
    conf_matrix = confusion_matrix(train_labels_argmax, train_predictions)

    # Storing the results
    results[model_type] = {
        'history': history,
        'report': report,
        'conf_matrix': conf_matrix
    }

    print(f"{model_type} - Classification Report:\n", classification_report(train_labels_argmax, train_predictions, target_names=emotion_labels))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title(f'{model_type} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_type} - Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} - Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Compare Models
model_comparison = {}
for model_type in models_to_compare:
    model_comparison[model_type] = {
        'Accuracy': results[model_type]['report']['accuracy'],
        'Macro Avg Precision': results[model_type]['report']['macro avg']['precision'],
        'Macro Avg Recall': results[model_type]['report']['macro avg']['recall'],
        'Macro Avg F1-Score': results[model_type]['report']['macro avg']['f1-score']
    }

comparison_df = pd.DataFrame(model_comparison).T
print("\nModel Comparison:\n", comparison_df)
best_model_type = comparison_df['Accuracy'].idxmax()   # model with the highest accuracy
print(f"Best Model based on Accuracy: {best_model_type}")
best_model = build_model(input_shape, num_classes, best_model_type)
best_model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
test_data_dir = 'dataset/test'
test_images, test_labels, _ = load_data(test_data_dir)

# Evaluating the best model on the test dataset
test_loss, test_accuracy = best_model.evaluate(test_images, test_labels)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report on test data
test_predictions = best_model.predict(test_images)
test_predictions = np.argmax(test_predictions, axis=1)
test_labels_argmax = np.argmax(test_labels, axis=1)

# Classification report and confusion matrix for test data
print("\nTest Classification Report:\n", classification_report(test_labels_argmax, test_predictions, target_names=emotion_labels))

# confusion matrix for test data
test_conf_matrix = confusion_matrix(test_labels_argmax, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Test Data - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
