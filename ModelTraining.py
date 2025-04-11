"""this program creates and trains a cnn model to classify which pose a user is doing.
the dataset (created by us) of 5 different poses, is passed through mediapipe to extract the pose landmarks,
so the model is trained on the pose landmarks instead of the images."""

import os
import cv2
import random
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def augment_image(image):
    # make list with both og and augmented images
    augmented_images = [image]
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Horizontal Random Flip
    if random.random() > 0.5:
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)

    # Random Rotation in degrees
    angle = random.uniform(-90, 90)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    augmented_images.append(rotated)

    # Brightness
    for factor in [0.7, 1.3]:
        bright = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        augmented_images.append(bright)

    # Random Zoom by 30%
    scale_factor = random.uniform(1.0 - 0.3, 1.0 + 0.3)
    new_size = (int(w * scale_factor), int(h * scale_factor))
    scaled = cv2.resize(image, new_size)
    start_x = max((new_size[0] - w) // 2, 0)
    start_y = max((new_size[1] - h) // 2, 0)
    cropped = scaled[start_y:start_y + h, start_x:start_x + w]
    augmented_images.append(cropped)

    # random contrast between 20%
    alpha = random.uniform(1.0 - 0.2, 1.0 + 0.2)
    contrast_adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    augmented_images.append(contrast_adjusted)

    # Resizing (extra) (h=32, w=32)
    resized = cv2.resize(image, (32, 32))
    augmented_images.append(resized)
    return augmented_images


#extracting pose landmarks from mediapiipe
def extract_landmarks(image, pose):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    else:
        return None


def process_dataset(image_folder, label):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    X, y = [], []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            augmented = augment_image(image)
            for img in augmented:
                lm = extract_landmarks(img, pose)
                if lm is not None:
                    X.append(lm)
                    y.append(label)
    return np.array(X), np.array(y)


def data():
    base_path = "DATA"
    X0, y0 = process_dataset(os.path.join(base_path, "UsainBolt"), label=0)
    X1, y1 = process_dataset(os.path.join(base_path, "contraposto"), label=1)
    X2, y2 = process_dataset(os.path.join(base_path, "kamehameha"), label=2)
    X3, y3 = process_dataset(os.path.join(base_path, "michael_jackson"), label=3)
    X4, y4 = process_dataset(os.path.join(base_path, "sailor_moon"), label=4)

    X_train = np.concatenate((X0, X1, X2, X3, X4), axis=0)
    y_train = np.concatenate((y0, y1, y2, y3, y4), axis=0)

    return X_train, y_train

def test_data():
    base_path = "TEST_DATA"

    x0, y0 = process_dataset(os.path.join(base_path, "UsainBolt"), label=0)
    x1, y1 = process_dataset(os.path.join(base_path, "contraposto"), label=1)
    x2, y2 = process_dataset(os.path.join(base_path, "kamehameha"), label=2)
    x3, y3 = process_dataset(os.path.join(base_path, "michael_jackson"), label=3)
    x4, y4 = process_dataset(os.path.join(base_path, "sailor_moon"), label=4)

    X_test = np.concatenate((x0, x1, x2, x3, x4), axis=0)
    y_test = np.concatenate((y0, y1, y2, y3, y4), axis=0)

    return X_test, y_test

# normalize landmark coordinates (for consistency)
def normalize_landmarks(landmarks):
    max_value = np.max(np.abs(landmarks))
    return landmarks / max_value if max_value != 0 else landmarks


def model(X):
    model = Sequential()
    # input layer
    model.add(Flatten(input_shape=(X.shape[1],)))

    # first hidden layer
    model.add(Dense(256))
    # LeakyReLU: Helps prevent dead neurons.
    model.add(LeakyReLU(alpha=0.1))
    # batch normalization stabilizes and speeds up training
    model.add(BatchNormalization())
    # dropout to prevent overfitting
    model.add(Dropout(0.4))

    # second hidden layer
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # third hidden layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    # output layer
    # softmax used for handling multiclass classification (5)
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def validate_model(X_train, X_test, y_train, y_test, model):
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=5,  # Stop if no improvement for 5 epochs
        restore_best_weights=True  # Restore the best model weights
    )

    history = model.fit(X_train, y_train,
                        epochs=40,
                        validation_data=(X_test, y_test),
                        callbacks = [early_stopping])

    # creating histogram to see how well things go
    plt.plot(history.history['accuracy'], label='Training Acc')
    plt.plot(history.history['val_accuracy'], label='Testing Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.show()

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}")

    # consufion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Usain", "contraposto", "kamehameha", "micheal_jackson", "sailor_moon"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()


def main():

    # Load and process the data
    X_train, y_train = data()
    X_test, y_test = test_data()

    # Build the model
    model_instance = model(X_train)

    # Train and validate the model
    validate_model(X_train, X_test, y_train, y_test, model_instance)

    # Save the model
    model_instance.save("model/test_model.h5")
    print("✅ Model saved as 'model/test_model.h5'")


if __name__ == "__main__":
    main()