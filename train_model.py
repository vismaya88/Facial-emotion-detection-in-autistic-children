import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import tensorflow as tf

train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Image data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Dataset visualization
def plot_dataset_distribution():
    class_counts = [len(os.listdir(os.path.join(train_dir, label))) for label in os.listdir(train_dir)]
    labels = os.listdir(train_dir)
    plt.figure(figsize=(10, 6))
    plt.bar(labels, class_counts, color='skyblue')
    plt.title('Dataset Distribution')
    plt.xlabel('Emotion Categories')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.show()

plot_dataset_distribution()

# Visualizing images by emotion
def visualize_images_by_emotion():
    emotion_labels = list(train_generator.class_indices.keys())
    plt.figure(figsize=(15, 10))
    for i, emotion in enumerate(emotion_labels[:6]):  
        img, label = next(train_generator)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img[0])
        plt.title(f"Emotion: {emotion}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_images_by_emotion()

# Base model
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze initial layers
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model_checkpoint = ModelCheckpoint('best_model_updated.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Training the model for initial pre-training
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=[model_checkpoint]
)

# Fine-tuning phase (Meta-Learning)
def fine_tune_model(model, train_generator, test_generator):
    # Unfreeze the layers after initial training
    for layer in model.layers[:100]:  
        layer.trainable = True
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    history_fine_tune = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator,
        callbacks=[model_checkpoint]
    )
    
    return history_fine_tune

history_fine_tune = fine_tune_model(model, train_generator, test_generator)

# Plotting training history
def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot the history after fine-tuning
plot_history(history_fine_tune)

# Evaluate the model
def evaluate_model():
    model = keras.models.load_model('best_model.keras')
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

evaluate_model()

