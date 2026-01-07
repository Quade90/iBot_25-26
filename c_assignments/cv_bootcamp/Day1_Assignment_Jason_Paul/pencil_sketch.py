import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def pencil_sketch(img_color, kernel_size=(21, 21)):
    """Converts a color image to a pencil sketch.

    Parameters:
        img_color (numpy.ndarray): Input color image in BGR format.
        kernel_size (tuple): Kernel size for Gaussian blur.

    Returns:
        Pencil sketch image in grayscale and colour format.
    """

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) # Convert to HSV color space
    img_hsv_h, img_hsv_s, img_hsv_v = cv2.split(img_hsv)

    img_hsv_s = (img_hsv_s * 0.6).astype(np.uint8) # Reduce saturation for grayscale effect

    img_blur = cv2.GaussianBlur(img_hsv_v, kernel_size, sigmaX=0, sigmaY=0) # Apply Gaussian blur
    img_sketch_v = cv2.divide(img_blur, img_hsv_v + 1, scale=256.0) # Blend hsv and inverted blur to create sketch
    img_sketch_v = np.clip(img_sketch_v, 0, 255) # Normalize to full 0-255 range

    img_sketch_gray = img_sketch_v.astype(np.uint8)


    img_sketch = cv2.merge((img_hsv_h, img_hsv_s, img_sketch_v)) # Merge channels to create grayscale sketch
    img_sketch = cv2.cvtColor(img_sketch, cv2.COLOR_HSV2BGR)

    return img_sketch, img_sketch_gray

def pencil_sketch_video(video, kernel_size=(21, 21)):
    """Converts each frame of a video to a pencil sketch, both colour and grayscale and saves it.

    Parameters:
        video (cv2.VideoCapture): Input video capture object.
        kernel_size (tuple): Kernel size for Gaussian blur.

    Returns:
        None
    """

    video = Path(video)
    video_handle = cv2.VideoCapture(str(video))

    if not video_handle.isOpened():
        raise ValueError("Could not open video file") # Handle invalid video file

    width = int(video_handle.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_handle.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30 # Default fps if unable to get from video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create outputs directory if it doesn't exist
    output_dir = Path("Outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{video.stem}_colour_sketch_video.mp4"
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height)) # Colour sketch video writer

    output_path = output_dir / f"{video.stem}_grayscale_sketch_video.mp4"
    video_writer_gray = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height)) # Grayscale sketch video writer

    # Process each frame
    while True:
        ret, frame = video_handle.read()
        if not ret:
            break
        img_sketch, img_sketch_gray = pencil_sketch(frame, kernel_size=kernel_size)
        video_writer.write(img_sketch)
        img_sketch_gray = cv2.cvtColor(img_sketch_gray, cv2.COLOR_GRAY2BGR)
        video_writer_gray.write(img_sketch_gray)

    video_handle.release()
    video_writer.release()
    video_writer_gray.release()

    print("Pencil sketch videos saved.")

def tandem_display(img_color, img_sketch):
    """Displays the original color image and the pencil sketch side by side.

    Parameters:
        img_color (numpy.ndarray): Original color image in BGR format.
        img_sketch (numpy.ndarray): Pencil sketch image in grayscale format.

    Returns:
        None
    """

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_sketch, cv2.COLOR_BGR2RGB))     
    plt.title("Pencil Sketch")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def kernel_size_input():
    """Accepts user input for kernel size and validates it.

    Parameters:
        None

    Returns:
        kernel_size (tuple): Validated kernel size for Gaussian blur.
    """
    
    try:
        kernel_side = int(input("Enter kernel side length: "))
    except ValueError:
        kernel_side = 21
        print("Invalid input. Using default kernel size (21, 21).") # Default kernel size if input is not integer
    if kernel_side < 0:
        kernel_size = (21, 21) # Default kernel size if negative number is provided
        print("Negative number provided. Using default kernel size (21, 21).")
    elif kernel_side % 2 == 0:
        kernel_size = (21, 21) # Default kernel size if even number is provided
        print("Even number provided. Using default kernel size (21, 21).")
    else:
        kernel_size = (kernel_side, kernel_side)
    return kernel_size

def get_file_type(path):
    """Determines if the file is a supported video based on its extension.

    Parameters:
        path (Path): Path object of the file.

    Returns:
        file_type (str): "video" if supported video, "missing" if file doesn't exist, "unknown" otherwise.  
    """
    vid_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

    if not path.exists():
        return "missing"

    if path.suffix.lower() in vid_exts:
        return "video"
    else:
        return "unknown"

    
class SketchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pencil Sketch GUI")
        self.root.geometry("1600x900")

        self.kernel = tk.IntVar(value=21)
        self.mode = tk.StringVar(value="color")

        self.img_color = None
        self.img_tk = None

        # ================= TOP: MODE SELECT =================
        mode_frame = tk.Frame(root)
        mode_frame.pack(pady=10)

        self.color_btn = tk.Button(
            mode_frame,
            text="Colour",
            font=("Arial", 12),
            width=12,
            command=lambda: self.set_mode("color")
        )
        self.color_btn.pack(side="left", padx=10)

        self.gray_btn = tk.Button(
            mode_frame,
            text="Grayscale",
            font=("Arial", 12),
            width=12,
            command=lambda: self.set_mode("gray")
        )
        self.gray_btn.pack(side="left", padx=10)

        # ================= LOAD IMAGE =================
        tk.Button(
            root,
            text="Load Image",
            font=("Arial", 12),
            command=self.load_image
        ).pack(pady=5)

        # ================= IMAGE DISPLAY =================
        self.image_frame = tk.Frame(root, width=1200, height=600)
        self.image_frame.pack(expand=True, fill="both", padx=20, pady=10)
        self.image_frame.pack_propagate(False)  # <- THIS IS THE MAGIC LINE

        self.panel = tk.Label(self.image_frame)
        self.panel.place(relx=0.5, rely=0.5, anchor="center")

        # ================= SLIDER =================
        tk.Scale(
            root,
            from_=3,
            to=103,
            resolution=2,
            orient=tk.HORIZONTAL,
            label="Blur Kernel",
            variable=self.kernel,
            command=self.update_sketch,
            length=600
        ).pack(pady=10)

        # ================= BOTTOM: SAVE BUTTONS =================
        save_frame = tk.Frame(root)
        save_frame.pack(pady=15)

        tk.Button(
            save_frame,
            text="Save As",
            font=("Arial", 12),
            width=12,
            command=self.save_image
        ).pack(side="left", padx=15)

        tk.Button(
            save_frame,
            text="Quick Save",
            font=("Arial", 12),
            width=12,
            command=self.quick_save
        ).pack(side="left", padx=15)


    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")]
        )
        if not path:
            return

        self.image_path = Path(path)   # ← ADD THIS
        self.img_color = cv2.imread(path)
        self.update_sketch()

    def update_sketch(self, event=None):
        if self.img_color is None:
            return

        k = self.kernel.get()
        if k % 2 == 0:
            k += 1

        sketch_color, sketch_gray = pencil_sketch(self.img_color, (k, k))

        if self.mode.get() == "gray":
            display = sketch_gray
            img = Image.fromarray(display)
        else:
            display = sketch_color
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

        w, h = img.size
        self.root.update_idletasks()

        max_w = self.image_frame.winfo_width()
        max_h = self.image_frame.winfo_height()

        if max_w < 10 or max_h < 10:
            return

        scale = min(max_w/w, max_h/h, 1)

        new_size = (int(w*scale), int(h*scale))
        img = img.resize(new_size, Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(img)

        self.panel.config(image=self.img_tk)
        self.panel.image = self.img_tk

    def save_image(self):
        if self.img_color is None:
            return

        stem = self.image_path.stem
        suffix = self.image_path.suffix

        default_name = (
            f"{stem}_grayscale_sketch{suffix}"
            if self.mode.get() == "gray"
            else f"{stem}_colour_sketch{suffix}"
        )

        path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=suffix,
            filetypes=[("Images", f"*{suffix}")]
        )
        if not path:
            return

        k = self.kernel.get()
        if k % 2 == 0:
            k += 1

        sketch_color, sketch_gray = pencil_sketch(self.img_color, (k, k))

        if self.mode.get() == "gray":
            cv2.imwrite(path, sketch_gray)
        else:
            cv2.imwrite(path, sketch_color)

    def quick_save(self):
        if self.img_color is None:
            return

        output_dir = Path("Outputs")
        output_dir.mkdir(exist_ok=True)

        k = self.kernel.get()
        if k % 2 == 0:
            k += 1

        stem = self.image_path.stem
        suffix = self.image_path.suffix
        mode = self.mode.get()

        if mode == "gray":
            filename = f"{stem}_grayscale_sketch{suffix}"
        else:
            filename = f"{stem}_colour_sketch{suffix}"

        path = output_dir / filename

        sketch_color, sketch_gray = pencil_sketch(self.img_color, (k, k))

        if mode == "gray":
            cv2.imwrite(str(path), sketch_gray)
        else:
            cv2.imwrite(str(path), sketch_color)

        print(f"Saved to {path}")

    def set_mode(self, mode):
        self.mode.set(mode)

        if mode == "color":
            self.color_btn.config(relief="sunken", bg="#cccccc")
            self.gray_btn.config(relief="raised", bg=self.root.cget("bg"))
        else:
            self.gray_btn.config(relief="sunken", bg="#cccccc")
            self.color_btn.config(relief="raised", bg=self.root.cget("bg"))

        self.update_sketch()



try:
    choice = input(
        "Process a video or an image? (video/image)\n"
        "Note: Videos must be placed inside the 'Tests' folder.\n> "
    ).strip().lower()

    if choice == "video":
        input_dir = Path("Tests")

        video_name = input("Enter video file name: ").strip()
        path = input_dir / video_name

        file_type = get_file_type(path)

        if file_type == "missing":
            raise FileNotFoundError("Video not found in Tests folder")

        if file_type != "video":
            raise ValueError("File is not a supported video format")

        kernel_size = kernel_size_input()
        print("Video processing...")
        pencil_sketch_video(path, kernel_size=kernel_size)

    elif choice == "image":
        if __name__ == "__main__":
            root = tk.Tk()
            root.geometry("1600x900")
            app = SketchGUI(root)
            root.mainloop()

    else:
        raise ValueError("Invalid choice. Enter 'video' or 'image'.")

except FileNotFoundError as e:
    print("File error:", e)

except ValueError as e:
    print("Input error:", e)

except cv2.error as e:
    print("OpenCV error:", e)

