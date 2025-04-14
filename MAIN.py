#this is the main file to run the program
#during the demo, only this needs to be run

from PoseRecognition import PoseRec
from removeBg import RemoveBackground
from Poster import Poster
import os
import serial
import time

def folder_handling(save_folder, og_folder):
    # get the latest image in the 'snapshots' folder
    save_folder = os.listdir(save_folder)
    if not save_folder:
        raise FileNotFoundError("No images found in folder")
    #sort files by modification time
    save_folder.sort(reverse=True)
    most_recent_file_path = os.path.join(og_folder, save_folder[0])
    return most_recent_file_path


def main():

    #fetching arduino input
    arduino = serial.Serial('COM7', 9600, timeout=1)
    time.sleep(2)
    print("Connected Arduino.")

    #Run PoseRecognition and get + classify the user pose
    model = "model/test_model.h5"
    pose_folder = "snapshots"
    no_bg_folder = "no_bg_images"
    poseRec = PoseRec(model, pose_folder)
    label = poseRec.run(serial_connection = arduino)
    most_recent_pose_path = folder_handling(pose_folder, og_folder= "snapshots")


    #use removeBg
    removeBg = RemoveBackground(most_recent_pose_path)
    black_image = removeBg.remove_background()
    person_cutout, fixed_colors = removeBg.final_trans_img(black_image, label)
    print(f"Background removed: {removeBg.show_final_img(fixed_colors)}")

    #make the Poster
    most_recent_no_bg_path = folder_handling(no_bg_folder, og_folder="no_bg_images")
    poster = Poster(most_recent_no_bg_path, label)
    poster.user_input(serial_connection=arduino)



if __name__ == "__main__":
    main()