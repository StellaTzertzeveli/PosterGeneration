import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time
import os

last_snapshot_time = 0
# sec between pic captures
cooldown_seconds = 50


model = load_model("model/test_model.h5")
pose_labels = ["Usain", "contraposto", "kamehameha", "micheal_jackson", "sailor_moon"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

save_folder = "snapshots"
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

def extract_live_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks), results
    else:
        return None, results

print("Start webcam. pls pose..")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, results = extract_live_landmarks(frame)

    if landmarks is not None and len(landmarks) == 99:
        input_data = np.expand_dims(landmarks, axis=0)
        prediction = model.predict(input_data)[0]
        predicted_label_idx = int(np.argmax(prediction))
        predicted_label = pose_labels[predicted_label_idx]
        confidence = prediction[predicted_label_idx]

        label_text = f"{predicted_label} ({confidence:.2f})"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.putText(frame, f"Detected points: {int(len(landmarks)/3)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        now = time.time()
        if confidence > 0.9 and (now - last_snapshot_time > cooldown_seconds):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(save_folder, f"{predicted_label}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"📸 {predicted_label} detected – saved as {filename}")
            last_snapshot_time = now
    else:
        cv2.putText(frame, "⛔ Incomplete or no pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Pose Detection", frame)
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # plt.imshow(frame_rgb)
    # plt.axis('off')
    # plt.title("Pose Detection")
    # plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()