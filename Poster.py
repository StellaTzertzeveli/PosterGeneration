#this class turns a picture into the poster
#a person without background is passed with its label,
#then it's matched to the corresponding background
#user can choose location & size of themselves and then write a title
#then the poster is saved/ printed

import os
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
import tkinter.simpledialog as dialog
import time
import subprocess


class Poster:
    # folder for finalized posters
    save_folder = "posters"

    # dictionary to map labels to background images
    bg_label = {
        "kamehameha": "dragonball.png",  # for kamehameha
        "contraposto": "museum.png",  # for contraposto
        "sailor_moon": "Sailor_moon.png",  # for sailor_moon
        "michael_jackson": "stage.png",  # for michael_jackson
        "usain_bolt": "usain.png"  # for usain_bolt
    }

    # dictionary to map labels to fonts
    font_label = {
        "kamehameha": "db.otf",  # for kamehameha
        "contraposto": "statue.ttf",  # for contraposto
        "sailor_moon": "sailor.ttf",  # for sailor_moon
        "michael_jackson": "michael.otf",  # for michael_jackson
        "usain_bolt": "usain.ttf"  # for usain_bolt
    }


    def __init__(self, person_cutout, img_label):

        # folder containing backgrounds & labeling
        self.background_folder = "backgrounds"
        # prepare the folder
        os.makedirs(self.save_folder, exist_ok=True)

        # prepared image with user's cutout
        self.person = Image.open(person_cutout).convert("RGBA")
        self.img_label = img_label

        self.width = self.person.size[1]
        self.height = self.person.size[0]

        # Default position (center) and scale for overlay
        self.pose_pos = [self.width // 2, self.height // 2]
        self.pose_scale = 1.0


    def background(self):
        #checks label of image to match background and returns it

        #is the label in dictionary
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
        whenever user changes things"""

        # Resize overlay based on current scale
        resized_overlay = overlay.resize(
            (int(self.width * self.pose_scale), int(self.height * self.pose_scale)),
            Image.Resampling.LANCZOS).convert("RGBA")

        # Paste resized pose onto background
        bg_copy = bg.copy()
        pose_x, pose_y = int(self.pose_pos[0]), int(self.pose_pos[1])
        bg_copy.paste(resized_overlay, (pose_x, pose_y), resized_overlay)

        # Convert the updated image to a format suitable for the Tkinter canvas
        tk_image = ImageTk.PhotoImage(bg_copy)
        canvas.image = tk_image  # Keep a reference to avoid garbage collection
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)


    def user_input(self, serial_connection=None):
        """create a GUI to make the poster.
        user can paste their cutout on the background, select size and position. """

        #make interactive display //Tkinter main window
        root = tk.Tk()
        root.title("Your fab poster! press SPACE to add a title. Press 1 to save it.")

        bg = self.background()
        scale_factor = 0.4  # Adjust this value to control the canvas size
        canvas_width = int(bg.width * scale_factor)
        canvas_height = int(bg.height * scale_factor)
        bg = bg.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        # Canvas for displaying images
        canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
        canvas.pack()

        #--------KEY events

        def move_left(event):
            self.pose_pos[0] = max(0, self.pose_pos[0] - 10)
            self.refresh_canvas(canvas, bg, self.person)

        def move_right(event):
            self.pose_pos[0] = min(bg.width - self.width * self.pose_scale, self.pose_pos[0])
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

        def add_text(event):
            #Prompt user for text input and add it as a title.
            text = tk.simpledialog.askstring("Input", "Enter title for the poster:")
            if text:
                new_bg = self.add_title(text, canvas_width, canvas_height, bg)
                print(f"Title added: {text}")
                self.refresh_canvas(canvas, new_bg, self.person)

        def save_poster_event(event):
            #Save the poster when '1' is pressed.
            self.save_poster(canvas)

        # Bind keys to functions
        #each of these has a corresponding key on the box
        root.bind("<Left>", move_left)
        root.bind("<Right>", move_right)
        root.bind("<Up>", move_up)
        root.bind("<Down>", move_down)
        root.bind("<equal>", zoom_in)  # Use '=' to zoom in
        root.bind("<minus>", zoom_out)  # Use '-' to zoom out
        root.bind("<t>", add_text)  # Use 't' to add text
        root.bind("<space>", add_text)  # Use 'space' to add text
        root.bind("1", save_poster_event)  # Use '1' to save the poster

        def poll_serial():
            # checks if there is data input from the box, and updates when there is

            if serial_connection and serial_connection.in_waiting > 0:

                if line == "left":
                    move_left(None)
                elif line == "right":
                    move_right(None)
                elif line == "up":
                    move_up(None)
                elif line == "down":
                    move_down(None)
                elif line == "green_pressed":
                    save_poster_event(None)
                elif line == "white_pressed":
                    add_text(None)
                elif line == "zoom_in":
                    zoom_in(None)
                elif line == "zoom_out":
                    zoom_out(None)

            # Schedule this function again after 100ms
            root.after(100, poll_serial)
            # Start polling Arduino input
        poll_serial()

        # Initial render of canvas
        self.refresh_canvas(canvas, bg, self.person)

        # Run the GUI loop
        root.mainloop()
        return canvas


    def add_title(self, text, cw, ch, bg):
        """Add a title to the poster at the top."""

        # Create a drawing context
        draw = ImageDraw.Draw(bg)

        #get different font depending on the label
        font_file = self.font_label[self.img_label]
        font_path = os.path.join("fonts", font_file)
        font = ImageFont.truetype(font_path, size=80)

        # Calculate text size using textbbox
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position the text at the top of the poster
        text_x = (cw - text_width) // 2
        text_y = (ch - text_height) // 7

        #text outline
        #basically putting the text twice in different colors
        offsets = [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)]
        for dx, dy in offsets:
            draw.text((text_x + dx, text_y + dy), text, font=font, fill="white")

        # Add the text to the poster
        draw.text((text_x, text_y), text, font=font, fill="black")
        return bg


    def save_poster(self, canvas):
        #Save the poster as PDF

        #paths for saving
        timestamp = int(time.time())
        ps_file = os.path.join(self.save_folder, f"temp_{timestamp}.ps")
        pdf_path = os.path.join(self.save_folder, f"{timestamp}_your_poster_as{self.img_label}.pdf")

        # save as postscript file
        try:
            canvas.postscript(file=ps_file, colormode="color")
            if not os.path.exists(ps_file) or os.path.getsize(ps_file) == 0:
                raise RuntimeError("Failed to create valid PS file")
        except Exception as e:
            print(f" couldn't make a ps file {e}")


        # convert to PDF using Ghostscript
        try:
            gs_path = r"C:\Program Files\gs\gs10.05.0\bin\gswin64c.exe"

            # Run Ghostscript directly (more reliable than ps2pdf)
            #rest of this block of code was generated by DeepSeek by feeding it about 15 different errors our code output
            result = subprocess.run(
                [
                    gs_path,
                    "-dBATCH",
                    "-dNOPAUSE",
                    "-sDEVICE=pdfwrite",
                    "-dPDFSETTINGS=/prepress",  # Higher quality settings
                    f"-sOutputFile={pdf_path}",
                    ps_file
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
                text=True
            )

            # Verify PDF was actually created
            if not os.path.exists(pdf_path):
                error_msg = result.stderr if result.stderr else "No error output"
                raise RuntimeError(f"Ghostscript ran but no PDF created. Error: {error_msg}")

            print(f"✅ Success! PDF saved to: {pdf_path}")
            print(f"PDF file size: {os.path.getsize(pdf_path) / 1024:.1f} KB")

        except subprocess.CalledProcessError as e:
            print(f"❌ Ghostscript failed with error code {e.returncode}")
            print(f"Error output: {e.stderr}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            # Clean up temporary PS file
            if os.path.exists(ps_file):
                try:
                    os.remove(ps_file)
                except Exception as e:
                    print(f"⚠ Couldn't delete temp file: {e}")