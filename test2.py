import cv2
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model('emotion_classification_model.h5')

# Define emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', 'Unknown', 'Unknown', 'Unknown']


# Capture video from camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess image
    resized_frame = cv2.resize(frame, (28, 28))
    grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    normalized_frame = grayscale_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 28, 28, 1))

    # Predict emotion
    predictions = model.predict(reshaped_frame)
    predicted_label = emotions[np.argmax(predictions)]

    # Draw predicted label on frame
    cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
