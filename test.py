import cv2
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('emotion_classification_model.h5')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
cap = cv2.VideoCapture(0)



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert color image to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize and normalize image
    resized_frame = cv2.resize(gray_frame, (48, 48))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 48, 48, 1))    # Preprocess image
    grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Predict emotion
    predictions = model.predict(reshaped_frame)
    predicted_label = emotions[np.argmax(predictions)]

    # Draw predicted label on frame
    cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destoryAllWindows()
