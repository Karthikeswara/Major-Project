import cv2
from keras.models import load_model
import numpy as np

model = load_model("fingers_model.h5")

# Start the camera
cap = cv2.VideoCapture(0)

fist_shown = False
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    # Resize the frame
    frame = cv2.resize(frame, (64,64))
    # Convert the frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Reshape the frame
    frame = frame.reshape(1, 64, 64, 1)

    # Use the trained model to predict the gesture
    prediction = model.predict(frame)
    # Get the index of the gesture with the highest probability
    gesture_index = np.argmax(prediction)

    if not fist_shown:
        # Check if the gesture is a fist
        if gesture_index == 0:
            print("Fist recognized! Proceed to next action")
            fist_shown = True
        else:
            # Display the live camera feed
            cv2.imshow("Camera", frame)
    else:
        if gesture_index == 1:
            print("Index finger recognized")
            fist_shown = False
        elif gesture_index == 2:
            print("Middle finger recognized")
            fist_shown = False
        elif gesture_index == 3:
            print("Index and middle fingers recognized")
            fist_shown = False
        else:
            # Display the live camera feed
            cv2.imshow("Camera", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        fist_shown = False
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()