#this class turns a picture into the poster
#a person without background is passed with its label,
#then it's matched to the corresponding background
#user can choose location & size of themselves and then write a title
#then the poster is saved/ printed

import os
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk #for the GUI
import tkinter.simpledialog as dialog

class Poster:
    # folder for finalized posters
    save_folder = "posters"

    bg_label = {
        "kamehameha": "backgrounds/dragonball.png",  # for kamehameha
        "contraposto": "backgrounds/museum.png",  # for contraposto
        "sailor_moon": "backgrounds/Sailor_moon.png",  # for sailor_moon
        "michael_jackson": "backgrounds/stage.png",  # for michael_jackson
        "usain_bolt": "backgrounds/usain.png"  # for usain_bolt
    }

    def __init__(self, person_cutout, img_label):

        # folder containing backgrounds & labeling
        self.background_folder = "backgrounds"
        # prepare the folder
        os.makedirs(self.save_folder, exist_ok=True)

        # prepared image with user's cutout
        self.person = Image.open(person_cutout)
        self.img_label = img_label

        self.width = self.person.size[1]
        self.height = self.person.size[0]

        # Default position (center) and scale for overlay
        self.pose_pos = [self.width // 2, self.height // 2]
        self.pose_scale = 1.0


    def background(self):
        #checks label of image to select the correct background and returns it

        #first see if label in dictionary
        if self.img_label not in self.bg_label:
            print("No background selected.")
            raise ValueError(f"Invalid label '{self.img_label}'. Must be one of: {list(self.bg_label.keys())}")

        bg_file = self.bg_label[self.img_label]
        background_path = os.path.join(self.background_folder, bg_file)

        if not os.path.exists(background_path):
            raise FileNotFoundError(f"Background image not found: {background_path}")

        #load and display the bg image
        return Image.open(background_path)

    def refresh_canvas(self, canvas, bg, overlay):
        """Refresh the canvas with the updated background and overlay.
        whenever user changes position/ zoom"""

        # Resize overlay based on current scale
        resized_overlay = overlay.resize(
            (int(self.width * self.pose_scale), int(self.height * self.pose_scale)),
            Image.ANTIALIAS
        )

        # Ensure overlay is in RGBA mode
        resized_overlay = resized_overlay.convert("RGBA")

        # Paste resized pose onto background
        bg_copy = bg.copy()
        pose_x, pose_y = int(self.pose_pos[0]), int(self.pose_pos[1])
        bg_copy.paste(resized_overlay, (pose_x, pose_y), resized_overlay)

        # Convert the updated image to a format suitable for the Tkinter canvas
        tk_image = ImageTk.PhotoImage(bg_copy)
        canvas.image = tk_image  # Keep a reference to avoid garbage collection
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)


    def user_input(self):
        """create a GUI to make the poster.
        user can paste their cutout on the background, select size and position. """

        #get background
        bg = self.background()

        #make interactive display //Tkinter main window
        root = tk.Tk()
        root.title("Your fab poster!")

        #canvas for displaying images
        canvas = tk.Canvas(root, width=bg.width, height=bg.height)
        canvas.pack()


        #key events

        def move_left(event):
            self.pose_pos[0] = max(0, self.pose_pos[0] - 10)
            self.refresh_canvas(canvas, bg, self.person)

        def move_right(event):
            self.pose_pos[0] = min(bg.width - self.width * self.pose_scale, self.pose_pos[0] + 10)
            self.refresh_canvas(canvas, bg, self.person)

        def move_up(event):
            self.pose_pos[1] = max(0, self.pose_pos[1] - 10)
            self.refresh_canvas(canvas, bg, self.person)

        def move_down(event):
            self.pose_pos[1] = min(bg.height - self.height * self.pose_scale, self.pose_pos[1] + 10)
            self.refresh_canvas(canvas, bg, self.person)

        def zoom_in(event):
            self.pose_scale = min(2.0, self.pose_scale + 0.1)  # Limit to 2x scale
            self.refresh_canvas(canvas, bg, self.person)

        def zoom_out(event):
            self.pose_scale = max(0.5, self.pose_scale - 0.1)  # Limit to 0.5x scale
            self.refresh_canvas(canvas, bg, self.person)

        # key events
        def add_text(event):
            """Prompt user for text input and add it as a title."""
            text = tk.simpledialog.askstring("Input", "Enter title for the poster:")
            if text:
                self.add_title(text)
                print(f"Title added: {text}")

        # Bind keys to functions
        root.bind("<Left>", move_left)
        root.bind("<Right>", move_right)
        root.bind("<Up>", move_up)
        root.bind("<Down>", move_down)
        root.bind("<plus>", zoom_in)  # Use '+' key for zoom in
        root.bind("<minus>", zoom_out)  # Use '-' key for zoom out
        root.bind("<t>", add_text) # Use 't' key to add text

        # Initial render of canvas
        self.refresh_canvas(canvas, bg, self.person)

        # Run the GUI loop
        root.mainloop()

    def add_title(self, text: str, font: str = "arial.ttf", size: int = 32):
        """Get user input (probably a name or title) and add it as a title to the top of the poster."""
        bg = self.background()  # Get the background image
        draw = ImageDraw.Draw(bg)

        # Load the font
        try:
            font = ImageFont.truetype(font, size)
        except IOError:
            font = ImageFont.load_default()  # Fallback to default font if not found

        # Calculate text position (centered at the top)
        text_width, text_height = draw.textsize(text, font=font)
        position = ((bg.width - text_width) // 2, 10)  # 10px padding from the top

        # Draw the text
        draw.text(position, text, fill="white", font=font)

        # Save the updated poster with the title
        poster_path = os.path.join(self.save_folder, "final_poster_with_title.png")
        bg.save(poster_path)
        print(f"✅ Poster with title saved as '{poster_path}'")


