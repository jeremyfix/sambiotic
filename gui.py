# coding: utf-8

# Standard imports
import sys
from collections import OrderedDict
import pathlib
import glob

# External imports
import numpy as np
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk

# from tkinter import Canvas, Button, Scale, HORIZONTAL
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sv_ttk


def build_model(modelname, device):
    # First solution, using the original image size 1024x1024
    # predictor = SAM2VideoPredictor.from_pretrained(f"facebook/{modelname}")

    configs = {
        "sam2.1-hiera-tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1-hiera-small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1-hiera-base-plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1-hiera-large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    checkpoints = {
        "sam2.1-hiera-tiny": "checkpoints/sam2.1_hiera_tiny.pt",
        "sam2.1-hiera-small": "checkpoints/sam2.1_hiera_small.pt",
        "sam2.1-hiera-base-plus": "checkpoints/sam2.1_hiera_base_plus.pt",
        "sam2.1-hiera-large": "checkpoints/sam2.1_hiera_large.pt",
    }

    cfg, ckpt = configs[modelname], checkpoints[modelname]

    predictor = SAM2ImagePredictor(build_sam2(cfg, ckpt, device))

    return predictor


class FileBrowser:

    def __init__(self, directory, device):
        # Look for all the images (PNG) into the directory
        self.files = []
        for ext in ["*.jpeg", "*.jpg", "*.png"]:
            self.files.extend(glob.glob(f"{directory}/{ext}"))

        self.directory = directory
        self.device = device

    def __getitem__(self, index):
        # Load the image and scale appropriately
        image = Image.open(self.files[index])
        image = np.array(image.convert("RGB"))

        return image

    def __len__(self):
        return len(self.files)


class BioticSegmentation:
    def __init__(self, directory, modelname):

        self.device = "cuda"  # or "cpu"
        self.offload_video_to_cpu = False
        self.offload_state_to_cpu = False

        self.point_size = 10
        self.last_mask_color = np.array([0, 0, 1])
        self.objects_colors = []
        self.positive_color = "red"
        self.negative_color = "green"
        self.drawing_box = False
        self.draw_mode = "box"  # Can be box or point
        self.point_mode = "positive"  # Can be "positive" or "negative"

        # Loads the SAM2 model
        self.predictor = build_model(modelname, self.device)

        self.output_path = pathlib.Path("./masks")
        self.image_idx = 0
        # Load the images
        self.load_images(directory)

        self.init_ui()

    def load_images(self, directory=None):

        refresh = False
        if directory is None:
            refresh = True
            directory = tkinter.filedialog.askdirectory(
                initialdir=self.image_dataset.directory,
                title="Select a directory of images",
            )
        self.output_path = pathlib.Path("./masks")

        self.image_dataset = FileBrowser(directory, self.device)

        # Initialize mask of zeros
        self.out_mask = [None] * len(self.image_dataset)
        self.current_annotation_id = [1] * len(self.image_dataset)
        self.init_processing()

        # Possibly refresh the UI
        if refresh:
            self.image_slider.configure(to=len(self.image_dataset) - 1)
            self.update_display()

    def init_processing(self):

        self.current_image = self.image_dataset[self.image_idx]

        # Parameters for the SAM2
        self.predictor.set_image(self.image_dataset[self.image_idx])

        # Prompts
        self.prompts = []
        for _ in range(len(self.image_dataset)):
            self.prompts.append({"positive": [], "negative": [], "box": None})

        # Temporary variable to hold the starting point of the box
        self.box1_x = None
        self.box1_y = None
        self.box2_x = None
        self.box2_y = None

    def compute_predictions(self):
        # Reset the state
        prev_mask = (
            self.out_mask[self.image_idx][...]
            == self.current_annotation_id[self.image_idx]
        )
        self.out_mask[self.image_idx][prev_mask] = 0

        # Define the prompts
        ann_obj_id = self.current_annotation_id[self.image_idx]

        image_prompts = {}

        # Get the prompts from the current slice
        positive_points = self.prompts[self.image_idx]["positive"]
        negative_points = self.prompts[self.image_idx]["negative"]
        box = self.prompts[self.image_idx]["box"]

        points = np.array(positive_points + negative_points, dtype=np.float32)
        if len(points) != 0:

            labels = np.array(
                ([1] * len(positive_points)) + ([0] * len(negative_points)),
                dtype=np.int32,
            )
            image_prompts["point_coords"] = points
            image_prompts["point_labels"] = labels

        if box is not None:
            x1, y1, x2, y2 = box
            image_prompts["box"] = [x1, y1, x2, y2]

        # And then perform the inference
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(
                **image_prompts,
                multimask_output=True,
            )
            # Sam2, by default, is multimask_output = True
            # It is suggested to pick the mask with the highest score
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind[0]]
        # Fill in the mask only on pixels previously annotated as
        # background
        bkgrd_mask = self.out_mask[self.image_idx] == 0
        self.out_mask[self.image_idx][
            np.logical_and(bkgrd_mask, masks == 1.0)
        ] = ann_obj_id

    def update_display(self, event=None):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        current_image = self.current_image.copy()
        current_image = current_image / 255.0

        # Overlay the mask over the image if available
        # First generate new colors if necessary
        if len(self.objects_colors) < self.current_annotation_id[self.image_idx]:
            self.objects_colors.append(np.random.rand(3))

        # Overlay the mask of all the objects already annotated
        overlaid = current_image.copy()

        for i in range(1, self.current_annotation_id[self.image_idx]):
            mask = self.out_mask[self.image_idx] == i
            overlaid[mask] = (
                0.5 * current_image[mask] + 0.5 * self.objects_colors[i - 1]
            )

        # And also the currently edited object
        mask = (
            self.out_mask[self.image_idx] == self.current_annotation_id[self.image_idx]
        )
        overlaid[mask] = 0.5 * current_image[mask] + 0.5 * self.last_mask_color

        img = Image.fromarray((overlaid * 255).astype(np.uint8))
        img = img.resize((canvas_width, canvas_height))  # , Image.ANTIALIAS)

        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        positive_points = self.prompts[self.image_idx]["positive"]
        negative_points = self.prompts[self.image_idx]["negative"]
        box = self.prompts[self.image_idx]["box"]

        if box is not None:
            x1, y1, x2, y2 = box
            x1, y1 = self.mask_coords_to_image_coords(x1, y1)
            x2, y2 = self.mask_coords_to_image_coords(x2, y2)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

        for x, y in positive_points:
            x, y = self.mask_coords_to_image_coords(x, y)
            self.canvas.create_oval(
                x - self.point_size // 2,
                y - self.point_size // 2,
                x + self.point_size // 2,
                y + self.point_size // 2,
                fill=self.positive_color,
            )
        for x, y in negative_points:
            x, y = self.mask_coords_to_image_coords(x, y)
            self.canvas.create_oval(
                x - self.point_size // 2,
                y - self.point_size // 2,
                x + self.point_size // 2,
                y + self.point_size // 2,
                fill=self.negative_color,
            )

    def image_coords_to_mask_coords(self, x, y):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x = x / canvas_width * self.current_image.shape[1]
        y = y / canvas_height * self.current_image.shape[0]
        return x, y

    def mask_coords_to_image_coords(self, x, y):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x = x / self.current_image.shape[1] * canvas_width
        y = y / self.current_image.shape[0] * canvas_height
        return x, y

    def add_point(self, xy):
        x, y = xy
        x, y = self.image_coords_to_mask_coords(x, y)
        if self.point_mode == "positive":
            self.prompts[self.image_idx]["positive"].append([x, y])
        else:
            self.prompts[self.image_idx]["negative"].append([x, y])

    def on_mouse_press(self, event):
        if self.draw_mode == "box":
            self.box1_x, self.box1_y = self.image_coords_to_mask_coords(
                event.x, event.y
            )
            self.drawing_box = True
        else:
            self.add_point((event.x, event.y))
        self.compute_predictions()
        self.update_display()

    def on_mouse_drag(self, event):
        if self.draw_mode == "box" and self.drawing_box:
            self.box2_x, self.box2_y = self.image_coords_to_mask_coords(
                event.x, event.y
            )
            self.prompts[self.image_idx]["box"] = [
                min(self.box1_x, self.box2_x),
                min(self.box1_y, self.box2_y),
                max(self.box1_x, self.box2_x),
                max(self.box1_y, self.box2_y),
            ]
            self.compute_predictions()
            self.update_display()

    def on_mouse_release(self, event):
        self.drawing_box = False
        # Only update the predictions on release when we draw a box
        if self.draw_mode == "box":
            self.compute_predictions()
            self.update_display()

    def on_keystroke(self, event):
        if event.char == "p":
            self.draw_mode = "point"
            self.point_mode = "positive"
            self.draw_mode_label.config(text="Draw Mode: Point (Positive)")
        elif event.char == "n":
            self.draw_mode = "point"
            self.point_mode = "negative"
            self.draw_mode_label.config(text="Draw Mode: Point (Negative)")
        elif event.char == "b":
            self.draw_mode = "box"
            self.draw_mode_label.config(text="Draw Mode: Box")
        # If the user presses the left or right arrow key
        # Go the next or previous slice and update the slider
        elif event.keysym == "Left":
            self.change_image(max(0, self.image_idx - 1))
            self.image_slider.set(self.image_idx)
            self.init_processing()
            self.update_display()
        elif event.keysym == "Right":
            self.change_image(min(len(self.image_dataset) - 1, self.image_idx + 1))
            self.image_slider.set(self.image_idx)
            self.init_processing()
            self.update_display()
        # On page up or page down, move 10 slices
        elif event.keysym == "Next":
            self.change_image(max(0, self.image_idx - 10))
            self.image_slider.set(self.image_idx)
            self.update_display()
        elif event.keysym == "Prior":
            self.change_image(min(len(self.image_dataset) - 1, self.image_idx + 10))
            self.image_slider.set(self.image_idx)
            self.update_display()
        # Reset the prompts of the current slice on "r" as well as the predicted mask
        elif event.char == "r":
            self.prompts[self.image_idx] = {"positive": [], "negative": [], "box": None}
            self.box1_x = None
            self.box1_y = None
            self.box2_x = None
            self.box2_y = None
            self.out_mask[self.image_idx][...] = 0
            self.current_annotation_id[self.image_idx] = 1
            self.update_display()
        # Quit on "q"
        elif event.char == "q":
            self.root.quit()
        elif event.keysym == "Return":
            self.current_annotation_id[self.image_idx] += 1
            self.prompts[self.image_idx]["positive"] = []
            self.prompts[self.image_idx]["negative"] = []
            self.prompts[self.image_idx]["box"] = None
            self.update_display()
        elif event.keysym == "BackSpace":
            # Delete the currently defined annotations
            self.prompts[self.image_idx]["positive"] = []
            self.prompts[self.image_idx]["negative"] = []
            self.prompts[self.image_idx]["box"] = None
            self.out_mask[self.image_idx][
                self.out_mask[self.image_idx]
                == self.current_annotation_id[self.image_idx]
            ] = 0
            self.update_display()
        # else:
        #     print(f"Unknown key {event.keysym} {event.char}")

    def change_image(self, value):
        self.image_idx = int(float(value))
        self.image_label.config(
            text=f"Image: {self.image_idx + 1}/{len(self.image_dataset)}"
        )

        if self.out_mask[self.image_idx] is None:
            self.out_mask[self.image_idx] = np.zeros(
                self.current_image.shape[:2], dtype=np.int32
            )
        self.current_image = self.image_dataset[self.image_idx]
        self.update_display()

    def save_masks(self):

        # Resize the mask down to 512 x 512
        mask = torch.nn.functional.interpolate(
            self.out_mask.unsqueeze(dim=1),
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )
        mask = mask.squeeze(dim=1) >= 0.0

    def init_ui(self):
        self.root = tk.Tk()
        self.root.title("X-Ray Segmentation")

        # Create a frame to hold both canvases
        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Canvas for the main image
        self.canvas = tk.Canvas(main_frame, width=512, height=512)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<Configure>", self.update_display)
        self.root.bind("<Key>", self.on_keystroke)

        # Frame for slice selection
        slice_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        slice_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.image_label = tk.Label(
            slice_frame,
            text=f"Image: {self.image_idx + 1}/{len(self.image_dataset)}",
        )
        self.image_label.pack(padx=5, pady=5)

        self.image_slider = ttk.Scale(
            slice_frame,
            from_=0,
            to=len(self.image_dataset) - 1,
            orient=tk.HORIZONTAL,
            command=self.change_image,
        )
        self.image_slider.set(self.image_idx)
        self.image_slider.pack(padx=5, pady=5)
        self.image_slider.bind(
            "<Motion>",
            lambda event: self.image_label.config(
                text=f"Image: {self.image_idx + 1}/{len(self.image_dataset)}"
            ),
        )

        # Frame for draw mode
        draw_mode_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        draw_mode_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.draw_mode_label = tk.Label(
            draw_mode_frame, text=f"Draw Mode: {self.draw_mode.capitalize()}"
        )
        self.draw_mode_label.pack(padx=5, pady=5)

        # Save button
        self.save_button = ttk.Button(
            self.root,
            text="Save Masks",
            command=lambda: self.save_masks_to_netcdf(),
        )
        self.save_button.pack(side=tk.TOP, pady=5)

        # Load button
        self.load_button = ttk.Button(
            self.root,
            text="Open directory",
            command=lambda: self.load_images(),
        )
        self.load_button.pack(side=tk.TOP, pady=5)

        # Help button
        self.help_button = ttk.Button(self.root, text="Help", command=self.show_help)
        self.help_button.pack(side=tk.TOP, pady=5)

        # Quit button
        self.quit_button = ttk.Button(self.root, text="Quit", command=self.root.quit)
        self.quit_button.pack(side=tk.TOP)

        # This is where the magic happens
        sv_ttk.set_theme("dark")

        self.root.protocol("WM_DELETE_WINDOW", sys.exit)
        self.root.mainloop()

        print("Exiting...")

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_text = tk.Text(help_window, wrap=tk.WORD, width=50, height=20)
        help_text.insert(tk.END, "Instructions:\n\n")
        help_text.insert(tk.END, "1. Use the slider to navigate through slices.\n")
        help_text.insert(
            tk.END, "2. Use the 'p' key to switch to positive point mode.\n"
        )
        help_text.insert(
            tk.END, "3. Use the 'n' key to switch to negative point mode.\n"
        )
        help_text.insert(tk.END, "4. Use the 'b' key to switch to box mode.\n")
        help_text.insert(tk.END, "5. Click and drag to draw a box or add points.\n")
        help_text.insert(
            tk.END,
            "6. Use the 'Propagate' button to propagate predictions both forward and backward given the prompts of the current slice.\n",
        )
        help_text.insert(tk.END, "7. Use the 'r' key to reset the current slice.\n")
        help_text.insert(tk.END, "8. Use the 'q' key to quit the application.\n")
        help_text.config(state=tk.DISABLED)
        help_text.pack(padx=10, pady=10)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <directory> <modelname")
        sys.exit(1)

    directory = sys.argv[1]
    modelname = sys.argv[2]
    BioticSegmentation(directory, modelname)
