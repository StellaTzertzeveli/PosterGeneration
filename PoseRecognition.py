import cv2
import numpy as np
import mediapipe as mp
import os
import time
from tensorflow.keras.models import load_model

#loading the model
model = load_model("model/test_model.h5")
pose_labels = ["Usain", "contraposto", "kamehameha", "micheal_jackson", "sailor_moon"]

#loading things for mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

#saving in snapshots
save_folder = "snapshots"
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
#stop if camera does not want to open
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

#normalize becasue we also do this in the model
def normalize_landmarks(landmarks):
    max_value = np.max(np.abs(landmarks))
    return landmarks / max_value if max_value != 0 else landmarks

#extracting landmark, seeing if more then 20 are visible otherwise, just disregard guess
def extract_landmarks(image, visibility_threshold=0.5):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks and len(results.pose_landmarks.landmark) >= 20:
        landmarks = []
        all_visible = True
        for lm in results.pose_landmarks.landmark:
            if lm.visibility < visibility_threshold:
                all_visible = False
            landmarks.extend([lm.x, lm.y, lm.z])

        if all_visible:
            landmarks = np.array(landmarks)
            normalized = normalize_landmarks(landmarks)
            return normalized, results

    return None, results

#detecting pose after 10 sec
def detect_pose():
    start_time = time.time()
    snapshot_frame = None
    label_text = ""
    landmarks = None
    results = None
    confidence = 0

    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            continue

        # keeping track of time
        clean_frame = frame.copy()
        seconds = time.time() - start_time
        countdown = int(10 - seconds)

        # extract landmarks continuously
        landmarks, results = extract_landmarks(frame)

        if landmarks is not None:
            input_data = np.expand_dims(landmarks, axis=0)
            prediction = model.predict(input_data)[0]
            predicted_label_idx = int(np.argmax(prediction))
            confidence = prediction[predicted_label_idx]

            if confidence > 0.9:
                label_text = f"{pose_labels[predicted_label_idx]} ({confidence:.2f})"
            else:
                label_text = "Confidence too low"
        else:
            label_text = "Not enough visible landmarks"

        if results and results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # show label text
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if confidence > 0.9 else (0, 0, 255), 2)

        if countdown > 3:
            cv2.putText(frame, f"Time left: {countdown}s", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 5)
        elif countdown > 0:
            cv2.putText(frame, str(countdown), (800, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 0), 12)

        elif countdown == 0:
            snapshot_frame = clean_frame # to use a cleanframe without mp overlay

            if landmarks is not None and confidence > 0.9:
                pose_predictions = np.expand_dims(landmarks, axis=0)
                pose_prediction = model.predict(pose_predictions)[0]

                predicted_label_idx = np.argmax(pose_prediction)
                label = pose_labels[predicted_label_idx]

                filename = f"{save_folder}/{label}{int(time.time())}.jpg"
                cv2.imwrite(filename, snapshot_frame)

                cv2.imshow("Pose Detection", frame)


                print(f"Pose '{label}' captured and saved")

                cv2.putText(snapshot_frame, "Pose detected!", (200, 600),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

            else:
                cv2.putText(snapshot_frame, "Image NOT saved: not enough landmarks or low confidence", (10, 600),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            cv2.imshow("Pose Detection", snapshot_frame)
            cv2.waitKey(2000)
            break

        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

def main():
    #print("Press SPACE to start pose detection with countdown. Press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # quit anytime
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # pressing space to start pose detection with countdown
        if key == ord(' '):
            detect_pose()
        else:
            cv2.putText(frame, "PRESS SPACE TO START: YOU HAVE 10 SECONDS DO A POSE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Pose Detection", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
