"""In PoseRecognition.py the model is utilised by the user in real time.
A camera window pops up and after pressing 'spacebar'
the user has about 10 seconds to get into one of the 5 poses and snap a picture.
Then the model classifies which pose the user did and
returns a picture with its corresponding label."""

import cv2
import numpy as np
import mediapipe as mp
import os
import time
from tensorflow.keras.models import load_model

class PoseRec:

    def __init__(self, model, save_folder):
        self.model = load_model(model)
        self.save_folder = save_folder

        #labels for the poses
        self.pose_labels = ["usain_bolt", "contraposto", "kamehameha", "michael_jackson", "sailor_moon"]

        # Initialize Mediapipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        # folder for saving snapshots
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Camera not accessible")


    def normalize_landmarks(self, landmarks):
        #normalize landmark coordinates (for consistency)
        max_value = np.max(np.abs(landmarks))
        return landmarks / max_value if max_value != 0 else landmarks


    def extract_landmarks(self, image):
        #extract landmarks, let the program continue only if more than 20 are visible
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        visibility_threshold = 0.5

        if results.pose_landmarks and len(results.pose_landmarks.landmark) >= 20:
            # Initialize a zero-filled array of size 99 cuz otherize its not working because we 2D detect but mediapipe is expecting 3D landmarks
            landmarks = np.zeros(99)
            all_visible = True

            for i, lm in enumerate(results.pose_landmarks.landmark):
                if lm.visibility < visibility_threshold:
                    all_visible = False
                if i < 33:  # Ensure we don't exceed the expected number of landmarks
                    landmarks[i * 3:i * 3 + 3] = [lm.x, lm.y, lm.z]

            if all_visible:
                landmarks = np.array(landmarks)
                normalized = self.normalize_landmarks(landmarks)
                return normalized, results

        return None, results

    def detect_pose(self):
        start_time = time.time()
        confidence = 0
        label = "No pose detected"

        while time.time() - start_time < 10:
            #once 10 seconds pass, capture a frame with webcam
            #that returns 2 values.
            #ret = boolean,if true, indicates if the frame was captured successfully
            #frame = the image captured, if ret = true
            ret, frame = self.cap.read()
            if not ret:
                #frame wasn't captured so skip the rest of loop and try again
                continue

            # keeping track of time
            #countdown works by taking the current time and subtracting it from the start time
            clean_frame = frame.copy()
            seconds = time.time() - start_time
            countdown = int(10 - seconds)

            landmarks, results = self.extract_landmarks(frame)

            if landmarks is not None:
                # expand the dimensions of landmarks to match the input shape of the model
                input_data = np.expand_dims(landmarks, axis=0)

                # predict the pose using the model
                prediction = self.model.predict(input_data)[0]
                predicted_label_idx = int(np.argmax(prediction))
                confidence = prediction[predicted_label_idx]

                if confidence > 0.9:
                    # confidence is high enough, show the label
                    label_text = f"{self.pose_labels[predicted_label_idx]} ({confidence:.2f})"
                    label = self.pose_labels[predicted_label_idx]
                else:
                    label_text = "Confidence too low"
            else:
                label_text = "Not enough visible landmarks"

            if results and results.pose_landmarks:
                # draw the pose landmarks on the frame
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # show label text
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if confidence > 0.9 else (0, 0, 255), 2)


            if countdown > 0:
                # show countdown timer
                cv2.putText(frame, f"Time left: {countdown}s", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 5)

            elif countdown > 0:
                # show countdown number
                cv2.putText(frame, str(countdown), (800, 700),
                            cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 0), 12)


            elif countdown == 0:
                # once countdown reaches 0, save the image
                snapshot_frame = clean_frame

                if landmarks is not None and confidence > 0.9:
                    # save the snapshot with the detected pose
                    pose_predictions = np.expand_dims(landmarks, axis=0)
                    pose_prediction = self.model.predict(pose_predictions)[0]

                    # get the predicted label
                    predicted_label_idx = np.argmax(pose_prediction)
                    label = self.pose_labels[predicted_label_idx]

                    # save the image with the time+label
                    filename = f"{self.save_folder}/{int(time.time())}{label}.jpg"
                    cv2.imwrite(filename, snapshot_frame)
                    print(f"Pose '{label}' captured and saved")

                    # show success message
                    cv2.putText(snapshot_frame, "Pose detected!", (200, 600),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
                else:
                    # show failure message
                    cv2.putText(snapshot_frame, "Image NOT saved: not enough landmarks or low confidence", (10, 600),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                # show the snapshot
                cv2.imshow("Pose Detection", snapshot_frame)
                cv2.waitKey(2000)


            cv2.imshow("Pose Detection", frame)


            if cv2.waitKey(1) == ord('q'):
                # quit the program if 'q' is pressed
                break

        return label

    def run(self):
        while self.cap.isOpened():
            # read a frame from the camera
            ret, frame = self.cap.read()
            if not ret:
                # frame wasn't captured so skip the rest of loop and try again
                break

            # quit anytime
            key = cv2.waitKey(1)
            if key == ord('q'):
                # quit the program if 'q' is pressed
                break
            if key == ord(' '):
                # pressing space to start pose detection with countdown
                final_label = self.detect_pose()
                return final_label
            else:
                # show instructions to start pose detection while space isnt pressed
                cv2.putText(frame, "PRESS SPACE TO START: YOU HAVE 10 SECONDS TO DO A POSE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Pose Detection", frame)

        self.cap.release()
        cv2.destroyAllWindows()