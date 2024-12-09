import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import sys
import os
import tensorflow as tf
import time


# Check if GPU is available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is being used.")
else:
    print("GPU not available, using CPU.")

# Adjusting the import path for the filters located in the 'scripts' folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Importing the filter functions from their respective files in the 'scripts' folder
from scripts.beard import apply_beard
from scripts.face import apply_face_overlay
from scripts.glass import apply_glasses_filter
from scripts.thugglass import apply_thugglasses_filter
from scripts.hat import apply_cowboy_hat, detect_faces
from scripts.moustache import apply_moustache  
from scripts.santa import apply_santa_hat

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller. """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Function to apply filters
def apply_filter(frame, filter_type):
    if filter_type == "Beard":
        return apply_beard(frame)  
    elif filter_type == "Face":
        return apply_face_overlay(frame)  
    elif filter_type == "Glass":
        return apply_glasses_filter(frame)  
    elif filter_type == "ThugGlass":
        return apply_thugglasses_filter(frame) 
    elif filter_type == "Cowboy Hat":
        faces = detect_faces(frame)  
        return apply_cowboy_hat(frame, faces)
    elif filter_type == "Moustache":
        return apply_moustache(frame)  
    elif filter_type == "Santa":
        return apply_santa_hat(frame)  
    elif filter_type == "None":
        return frame
    return frame

# Class for the application
class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Snapchat-like Camera App")
        self.root.resizable(False, False)

        self.canvas = ctk.CTkCanvas(root, width=640, height=480)
        self.canvas.pack(padx=10, pady=10)

         # Add FPS label to display FPS on GUI
        self.fps_label = ctk.CTkLabel(root, text="FPS: 0", font=("Arial", 14))
        self.fps_label.pack(pady=10)
        # Define filters with icons
        self.filters = {
            "None": {"label": "", "function": lambda frame: apply_filter(frame, "None"), "image": resource_path("icons/none.png")},
            "Beard": {"label": "", "function": lambda frame: apply_filter(frame, "Beard"), "image": resource_path("filters/beard/white beard.png")},
            "Face": {"label": "", "function": lambda frame: apply_filter(frame, "Face"), "image": resource_path("filters/face/mrbean.png")},
            "Glass": {"label": "", "function": lambda frame: apply_filter(frame, "Glass"), "image": resource_path("filters/glasses/glass.png")},
            "ThugGlass": {"label": "", "function": lambda frame: apply_filter(frame, "ThugGlass"), "image": resource_path("filters/glasses/thugglass.png")},
            "Cowboy Hat": {"label": "", "function": lambda frame: apply_filter(frame, "Cowboy Hat"), "image": resource_path("filters/hats/realhat.png")},
            "Moustache": {"label": "", "function": lambda frame: apply_filter(frame, "Moustache"), "image": resource_path("filters/moustache/light.png")},
            "Santa": {"label": "", "function": lambda frame: apply_filter(frame, "Santa"), "image": resource_path("filters/santa/santahat.png")},
        }

        self.filter_names = list(self.filters.keys())
        self.current_filter_index = 0

        self.filter_icons = {}
        for filter_name, filter_data in self.filters.items():
            icon_image = Image.open(filter_data["image"]).resize((40, 40))
            self.filter_icons[filter_name] = ctk.CTkImage(light_image=icon_image)

        self.prev_icon = ctk.CTkImage(light_image=Image.open(resource_path("icons/left2.png")).resize((40, 40)))

        self.prev_button = ctk.CTkButton(root, text="", command=self.show_previous_filter, width=80, height=40, image=self.prev_icon)
        self.prev_button.pack(side="left", padx=10)

        self.filter_buttons = []
        for filter_name, filter_data in self.filters.items():
            filter_button = ctk.CTkButton(
                root, text="", image=self.filter_icons[filter_name],
                command=lambda f=filter_name: self.set_filter(f), width=30, height=30,
                corner_radius=100
            )
            self.filter_buttons.append(filter_button)

        for button in self.filter_buttons:
            button.pack(side="left", padx=10, pady=10)

        self.cap = cv2.VideoCapture(0)
        self.video_width = 640
        self.video_height = 480

        self.next_icon = ctk.CTkImage(light_image=Image.open(resource_path("icons/next2.png")).resize((40, 40)))

        self.next_button = ctk.CTkButton(root, text="", command=self.show_next_filter, width=80, height=40, image=self.next_icon)
        self.next_button.pack(side="left", padx=10)

        self.prev_time = time.time()  # Variable for FPS calculation
        self.update_frame()

    def set_filter(self, filter_type):
        self.current_filter_index = self.filter_names.index(filter_type)

    def update_frame(self):
        ret, frame = self.cap.read()
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        # Update FPS label
        self.fps_label.configure(text=f"FPS: {int(fps)}")
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.video_width, self.video_height))
            filter_name = self.filter_names[self.current_filter_index]
            filtered_frame = self.filters[filter_name]["function"](frame)

            if filter_name == "Gray":
                filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)

            self.canvas.create_image(0, 0, image=imgtk, anchor="nw")
            self.canvas.imgtk = imgtk

        self.root.after(30, self.update_frame)

    def show_previous_filter(self):
        self.current_filter_index = (self.current_filter_index - 1) % len(self.filters)
        print(f"Previous Filter: {self.filter_names[self.current_filter_index]}")  

    def show_next_filter(self):
        self.current_filter_index = (self.current_filter_index + 1) % len(self.filters)
        print(f"Next Filter: {self.filter_names[self.current_filter_index]}")  

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
