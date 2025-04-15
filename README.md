# PosterGeneration
group 16

This is a program where a person can create a fun poster of themselves to boost their confidence!
The project consists of 4 main parts.

1. In ModelTraining.py we are training a sequential model, Neural Network to classify which pose a user is doing.
The dataset (created by us) of 5 different poses, made by around 60 different people, is passed through mediapipe to extract the pose landmarks,
and then the model is trained on the pose landmarks instead of the images.

2. In PoseRecognition.py the model is utilised by the user in real time. A camera window  pops up and after pressing 'spacebar' the user has about 10 seconds to get into one of the 5 poses and snap a picture. Then the model classifies which pose the user did and returns a picture with its corresponding label.

3. The class RemoveBg is utilized, where the snapshot taken is turned into an image with a transparent background with the libraries rembg and cv2.

4. The class Poster makes everything come together. A Graphical User Interface is created. The matching background is used, and the pose is pasted on top. Then the user can choose the location & size of their figure and write a title. Then the poster is saved/ printed.
