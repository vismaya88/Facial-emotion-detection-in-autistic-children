import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
import cv2

model = tf.keras.models.load_model("best_model.keras")
test_dir = "dataset/test"

# Data generator for the test dataset
test_datagen = ImageDataGenerator(rescale=1.0 / 255)  
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())  
report = classification_report(y_true, y_pred, target_names=class_labels)
print("Classification Report:")
print(report)
accuracy = np.sum(y_true == y_pred) / len(y_true)
print(f"Accuracy: {accuracy:.2f}")

# Real-time emotion detection using OpenCV
def predict_emotion_from_camera():
    class_labels = list(test_generator.class_indices.keys())  
    cap = cv2.VideoCapture(0)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  
        img = np.expand_dims(img, axis=0) / 255.0  

        # Predict emotion
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        predicted_class_name = class_labels[predicted_class]
        confidence = np.max(predictions) * 100  

        # Display the result on the frame
        cv2.putText(frame, f'{predicted_class_name} ({confidence:.2f}%)',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detection', frame)
        # Exit the qloop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
predict_emotion_from_camera()