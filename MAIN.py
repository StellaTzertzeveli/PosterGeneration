#this is the main file to run the program
#during the demo, only this needs to be run

from PoseRecognition import PoseRec
from removeBg import RemoveBackground
from Poster import Poster

removeBg = RemoveBackground()
# poster = Poster()


def main():
    # Step 1: Run PoseRecognition
    model = "model/test_model.h5"
    save_folder = "posters"
    poseRec = PoseRec(model, save_folder)
    label, pose_output = poseRec.run()


    # Step 2: use removeBg
    black_image = removeBg.remove_background(pose_output)
    bgra, trans_image = removeBg.final_trans_img(black_image)
    print(f"Background removed: {removeBg.show_final_img(bgra)}")

    # Step 3: make the Poster
    poster = Poster(trans_image, label)
    background = poster.background()
    while(True):
        canvas = poster.user_input()
        title_text = poster.add_title(canvas)
        poster.refresh_canvas(canvas, background, bgra)


if __name__ == "__main__":
    main()