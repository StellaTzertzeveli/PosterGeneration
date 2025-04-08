import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


def augment_image(image):
    augmented_images = [image]  # include original
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # Flip
    augmented_images.append(cv2.flip(image, 1))

    # Rotate
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)

    # Brightness
    for factor in [0.7, 1.3]:
        bright = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        augmented_images.append(bright)

    # Zoom
    scale_factor = 1.2
    new_size = (int(w * scale_factor), int(h * scale_factor))
    scaled = cv2.resize(image, new_size)
    start_x = (new_size[0] - w) // 2
    start_y = (new_size[1] - h) // 2
    cropped = scaled[start_y:start_y + h, start_x:start_x + w]
    augmented_images.append(cropped)

    return augmented_images


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

#extracting pose landmarks from mediapiipe
def extract_landmarks(image):
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
    X, y = [], []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            augmented = augment_image(image)
            for img in augmented:
                lm = extract_landmarks(img)
                if lm is not None:
                    X.append(lm)
                    y.append(label)
    return np.array(X), np.array(y)

# pose_folder = "/Users/nathanschuijt/PycharmProjects/Backgroudnremoval test/pose_images/usain_bolt"           # Map met correcte poses (label 0)
# non_pose_folder = "/Users/nathanschuijt/PycharmProjects/Backgroudnremoval test/pose_images/t_pose"   # Map met foute/random poses (label 1)

base_path = "/Users/nathanschuijt/PycharmProjects/Backgroudnremoval test/pose_images/"
X0, y0 = process_dataset(os.path.join(base_path, "usain_bolt"), label=0)
X1, y1 = process_dataset(os.path.join(base_path, "contraposto"), label=2)
X2, y2 = process_dataset(os.path.join(base_path, "kamehameha"), label=3)
X3, y3 = process_dataset(os.path.join(base_path, "michael_jackson"), label=4)
X4, y4 = process_dataset(os.path.join(base_path, "sailor_moon"), label=5)

# X_pose, y_pose = process_dataset(pose_folder, label=0)
# X_nonpose, y_nonpose = process_dataset(non_pose_folder, label=1)

X = np.concatenate((X0, X1, X2, X3, X4), axis=0)
y = np.concatenate((y0, y1, y2, y3, y4), axis=0)


# X = np.concatenate((X_pose, X_nonpose), axis=0)
# y = np.concatenate((y_pose, y_nonpose), axis=0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


model = Sequential()
model.add(Flatten(input_shape=(X.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))


#creating histogram to see how well
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

#consufion matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
labels = ["Usain", "contraposto", "kamehameha", "micheal_jackson", "sailor_moon"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()


model.save("/Users/nathanschuijt/PycharmProjects/Backgroudnremoval test/saves/pose_model.h5")
print("✅ Model opgeslagen als pose_model.h5")