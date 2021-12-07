import cv2
from tensorflow.keras import models, layers
import numpy as np
import os

LABELS = os.listdir('CK+48')


def get_label(k):
    return LABELS[np.argmax(k)]


def get_model(checkpoint='checkpoint/model', N_LABELS=7):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), input_shape=(48, 48, 1),
                      activation='elu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='elu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='elu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(128, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(N_LABELS, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.load_weights(checkpoint)
    return model


model = get_model()
vid = cv2.VideoCapture(1)
cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(
    'haarcascade_frontalcatface_extended.xml')

while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if frame is None:
        continue

    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(frame_new, 1.03, 3)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = np.array(face).reshape(48, 48, 1)
        p = model.predict(np.array([face]))[0]
        label = get_label(p)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255))
        frame = cv2.putText(frame, label, (x, y+h+32),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

    # Display the resulting frame
    cv2.imshow('Camera', frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
