import numpy as np
import cv2
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

latest_circled = None
latest_stats = None # stats for file
latest_stats_summary = None # stats for window

def preprocess_image(img):
    """Convert image to grayscale and apply median blur to reduce noise.

    Parameters:
        img (numpy.ndarray): Input color image.

    Returns:
        numpy.ndarray: Preprocessed grayscaled and blurred image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    blur = cv2.medianBlur(gray, 5)
    return blur   

def detect_circles(img, blur, base_params, param2=None, min_r=None, max_r=None):
    """Detect and label circles in the image using Hough Circle Transform. Categorizes circles by size.
    
    Parameters:
        img (numpy.ndarray): Original color image.
        blur (numpy.ndarray): Preprocessed grayscaled and blurred image.
        base_params (dict): Base parameters for Hough Circle Transform.
        param2 (int, optional): Accumulator threshold for circle detection.
        min_r (int, optional): Minimum circle radius.
        max_r (int, optional): Maximum circle radius.

    Returns:
        numpy.ndarray: Image with detected circles labeled.
        int: Number of detected circles.
        list: List of detected circles with their parameters.
        """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Invalid input image")

    if blur is None or len(blur.shape) != 2:
        raise ValueError("Blurred image must be grayscale")
    
    HOUGH_PARAMS = base_params.copy()

    if param2 is not None:
        HOUGH_PARAMS["param2"] = param2
    if min_r is not None:
        HOUGH_PARAMS["minRadius"] = min_r
    if max_r is not None:
        HOUGH_PARAMS["maxRadius"] = max_r


    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    HOUGH_PARAMS["dp"],
    HOUGH_PARAMS["minDist"],
    param1=HOUGH_PARAMS["param1"],
    param2=HOUGH_PARAMS["param2"],
    minRadius=HOUGH_PARAMS["minRadius"],
    maxRadius=HOUGH_PARAMS["maxRadius"]
    )

    circled_img = img.copy()

    # Calculate size thresholds for circle categorization
    h, w = img.shape[:2]

    min_dim = min(h, w)

    min_radius = int(min_dim * 0.05)
    max_radius = int(min_dim * 0.25)

    colour_range = max_radius - min_radius

    # Draw detected circles and labels
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for count, i in enumerate(circles[0, :]):
            tx = i[0] + int(i[2] / np.sqrt(2)) + 10
            ty = i[1] + int(i[2] / np.sqrt(2)) + 5

            if i[2] > min_radius:
                color = (255, 0, 0)  # Blue for small circles
            if i[2] > colour_range * 0.33 + min_radius:
                color = (0, 255, 255)  # Yellow for medium circles
            if i[2] > colour_range * 0.66 + min_radius:
                color = (0, 0, 255)  # Red for large circles
            
            cv2.circle(circled_img, (i[0], i[1]), i[2], color, 2)
            cv2.circle(circled_img, (i[0], i[1]), 2, (0, 0, 255), -1)
            cv2.line(circled_img, (i[0] + int(i[2]/np.sqrt(2)), i[1] + int(i[2]/np.sqrt(2))), (i[0] + int(i[2]/np.sqrt(2)) + 10, i[1] + int(i[2]/np.sqrt(2)) + 10), color, 3)
            cv2.line(circled_img, (i[0] + int(i[2]/np.sqrt(2)) + 10, i[1] + int(i[2]/np.sqrt(2)) + 10), (i[0] + int(i[2]/np.sqrt(2)) + 55, i[1] + int(i[2]/np.sqrt(2)) + 10), color, 2)
            cv2.putText(circled_img, f"({count + 1}){i[2]}px", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)    

        return circled_img, len(circles[0]), circles
    else:
        return img, 0, [[]]

def auto_tune_hough_params(gray):
    """Auto-tune Hough Circle Transform parameters based on the input image.

    Parameters:
        gray (numpy.ndarray): Preprocessed grayscaled image.
        
    Returns:
        dict: Tuned parameters for Hough Circle Transform.
    """
    dp = 1.2
    h, w = gray.shape
    min_dim = min(h, w)

    param1 = 120

    min_radius = int(min_dim * 0.05)
    max_radius = int(min_dim * 0.25)
    min_dist = int(max_radius * 0.5)

    for param2 in range(35, 80, 5):
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp,
            min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is not None:
            count = len(circles[0])

            if count > 50:
                continue  # reject garbage

            if 1 <= count <= 15:
                return {
                    "dp": dp,
                    "minDist": min_dist,
                    "param1": param1,
                    "param2": param2,
                    "minRadius": min_radius,
                    "maxRadius": max_radius
                }

    return {
        "dp": dp,
        "minDist": min_dist,
        "param1": param1,
        "param2": 35,
        "minRadius": min_radius,
        "maxRadius": max_radius
    }

def update_image(*args):
    """Update the displayed image based on current slider values.
    
    Parameters:
        *args: Additional arguments (not used).
    
    Returns:
        None
    """
    if img is None or blur is None:
        return

    p2 = param2_var.get()
    min_r = minr_var.get()
    max_r = maxr_var.get()

    # Enforce constraints
    if min_r < 1:
        min_r = 1
        minr_var.set(min_r)

    if max_r < 2:
        max_r = 2
        maxr_var.set(max_r)

    min_r = max(1, min_r)
    max_r = max(2, max_r)

    if min_r >= max_r:
        max_r = min_r + 1

    minr_var.set(min_r)
    maxr_var.set(max_r)

    display, count, circles = detect_circles(
        img,
        blur,
        BASE_PARAMS,
        param2=p2,
        min_r=min_r,
        max_r=max_r
    )

    global latest_circled, latest_stats
    latest_circled = display

    if count > 0:
        # window stats
        radii = [int(c[2]) for c in circles[0]]

        latest_stats_summary = (
            f"Circles detected: {count}\n"
            f"Min radius: {min(radii)} px\n"
            f"Max radius: {max(radii)} px\n"
            f"Avg radius: {sum(radii)/len(radii):.2f} px\n"
        )

        stats_var.set(latest_stats_summary)

        colour_range = max_r - min_r

        # file stats
        lines = [f"Circles detected: {count}\n\n"]

        for idx, c in enumerate(circles[0], start=1):
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            if r > min_r and r <= (colour_range * 0.33 + min_r):
                lines.append(f"{idx}. Small circle: x={x}, y={y}, radius={r}px\n")
            elif r > (colour_range * 0.33 + min_r) and r <= (colour_range * 0.66 + min_r):
                lines.append(f"{idx}. Medium circle: x={x}, y={y}, radius={r}px\n")
            else:
                lines.append(f"{idx}. Large circle: x={x}, y={y}, radius={r}px\n")
        
        lines.append("\n")
        lines.append(f"Min radius: {min(radii)} px\n")
        lines.append(f"Max radius: {max(radii)} px\n")
        lines.append(f"Avg radius: {sum(radii)/len(radii):.2f} px\n")
        lines.append("\n")
        lines.append(f"Parameters used:\n")
        lines.append(f"  dp: {BASE_PARAMS['dp']}\n")
        lines.append(f"  minDist: {BASE_PARAMS['minDist']}\n")
        lines.append(f"  param1: {BASE_PARAMS['param1']}\n")
        lines.append(f"  param2: {p2}\n")
        lines.append(f"  minRadius: {min_r}\n")
        lines.append(f"  maxRadius: {max_r}\n")

        latest_stats = "".join(lines)
    else:
        latest_stats = "No circles detected\n"
        stats_var.set(latest_stats)


    # Resize for display if image too large
    h, w = display.shape[:2]
    MAX_DISPLAY_H = 500

    if h > MAX_DISPLAY_H:
        scale = MAX_DISPLAY_H / h
        display = cv2.resize(
            display,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    disp_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    tk_img2 = ImageTk.PhotoImage(Image.fromarray(disp_rgb))
    img_label.configure(image=tk_img2)
    img_label.image = tk_img2

def quick_save():
    """Quickly save the latest circled image and stats to the outputs directory.
    
    Parameters:
        None
    
    Returns:
        None
    """
    if latest_circled is None:
        return

    img_name = path.stem

    img_out = OUTPUTS_DIR / f"{img_name}_circled.png"
    txt_out = OUTPUTS_DIR / f"{img_name}_stats.txt"

    cv2.imwrite(str(img_out), latest_circled)

    with open(txt_out, "w") as f:
        f.write(latest_stats)

def save_as():
    """Save the latest circled image and stats to a user-specified location.
    
    Parameters:
        None
    
    Returns:
        None
    """
    if latest_circled is None:
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png")]
    )

    if not file_path:
        return

    img_path = Path(file_path)
    txt_path = img_path.with_suffix(".txt")

    cv2.imwrite(str(img_path), latest_circled)

    with open(txt_path, "w") as f:
        f.write(latest_stats)

def load_image():
    """Load an image file, preprocess it, auto-tune parameters, and update the GUI.
    
    Parameters:
        None

    Returns:
        None
    """
    global img, blur, BASE_PARAMS, MAX_R, path, latest_circled, latest_stats

    file_path = filedialog.askopenfilename(
        title="Select image",
        initialdir=TESTS_DIR,
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )

    if not file_path:
        return

    path = Path(file_path)

    img = cv2.imread(str(path))
    if img is None:
        return

    # obtain dimensions of preprocessed image
    blur = preprocess_image(img)
    BASE_PARAMS = auto_tune_hough_params(blur)

    min_dim = min(img.shape[:2])
    MAX_R = int(min_dim * 0.3)

    img_h, img_w = img.shape[:2]

    MAX_DISPLAY_H = 500
    display_h = min(img_h, MAX_DISPLAY_H)

    root.update_idletasks()

    # slider ranges
    minr_scale.config(to=MAX_R)
    maxr_scale.config(to=MAX_R)

    # Force sliders to auto-tuned values
    param2_var.set(BASE_PARAMS["param2"])
    minr_var.set(BASE_PARAMS["minRadius"])
    maxr_var.set(BASE_PARAMS["maxRadius"])

    # Clear old outputs
    latest_circled = None
    latest_stats = None
    stats_var.set("Auto-tuned parameters applied.")

    # Enable sliders FIRST
    param2_scale.config(state="normal")
    minr_scale.config(state="normal")
    maxr_scale.config(state="normal")

    # Force Tkinter to calculate real widget sizes
    root.update_idletasks()

    # Clamp display height
    display_h = min(img.shape[0], 400)

    # Measure actual control height (sliders + buttons)
    controls_h = frame_ctrl.winfo_reqheight()

    # Final window size
    window_width = img.shape[1] + 40
    window_height = display_h + controls_h + 120

    # Center horizontally, 40px from top
    screen_w = root.winfo_screenwidth()
    x = (screen_w - window_width) // 2
    y = 40

    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.minsize(window_width, window_height)

    stats_label.config(wraplength=window_width - 40)

    update_image()

def process_video(video_path, output_dir):
    """Process a video file to detect circles in each frame and save the output video."
    
    Parameters:
        video_path (Path): Path to the input video file.
        output_dir (Path): Directory to save the output video.
    
    Returns:
        None
    """

    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir.mkdir(exist_ok=True)
    video_out = output_dir / f"{video_path.stem}_circled.mp4"

    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    base_params = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blur = preprocess_image(frame)

        # Auto-tune once in the beginning
        if base_params is None:
            base_params = auto_tune_hough_params(blur)

        circled, _, _ = detect_circles(
            frame,
            blur,
            base_params
        )

        writer.write(circled)

    cap.release()
    writer.release()


def is_video_file(path: Path) -> bool:
    """Check if the given path is a valid video file that can be opened.
    
    Parameters:
        path (Path): Path to the video file.
        
    Returns:
        bool: True if the file is a valid video, False otherwise.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False

    # attempt to read a frame
    ret, _ = cap.read()
    cap.release()
    return ret

def load_video():
    """Load a video file, process it to detect circles in each frame, and save the output video.
    
    Parameters:
        None
    
    Returns:
        None
    """
    file_path = filedialog.askopenfilename(
        title="Select video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )

    if not file_path:
        return

    path = Path(file_path)

    if not is_video_file(path):
        stats_var.set("Selected file is not a valid video.")
        return

    stats_var.set("Processing video...")
    root.update_idletasks()

    process_video(path, OUTPUTS_DIR)

    stats_var.set("Video processed and saved.")

def legend_row(parent, color, text):
    """Create a legend row with a colored box and text label.
    
    Parameters:
        parent (tk.Widget): Parent widget to contain the legend row.
        color (str): Color of the box.
        text (str): Text label for the legend.
    
    Returns:
        None
    """
    row = tk.Frame(parent)
    row.pack(anchor="w", pady=2)

    box = tk.Canvas(row, width=15, height=15, highlightthickness=0)
    box.create_rectangle(0, 0, 15, 15, fill=color, outline="black")
    box.pack(side=tk.LEFT)

    tk.Label(row, text=text, padx=5).pack(side=tk.LEFT)

def auto_tune_button():
    """Button to auto-tune Hough Circle Transform parameters based on the loaded image.
    
    Parameters:
        None
    
    Returns:
        None
    """
    global BASE_PARAMS

    if blur is None:
        stats_var.set("Load an image first.")
        return

    BASE_PARAMS = auto_tune_hough_params(blur)

    # Update sliders
    param2_var.set(BASE_PARAMS["param2"])
    minr_var.set(BASE_PARAMS["minRadius"])
    maxr_var.set(BASE_PARAMS["maxRadius"])

    stats_var.set("Parameters auto-tuned.")

    update_image()

# Set up directories
TESTS_DIR = Path("tests")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

root = tk.Tk()

root.geometry("800x600+200+80")

img = None
blur = None
BASE_PARAMS = None
path = None

root.title("Hough Circle Tuning")

# Main frames
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

frame_img = tk.Frame(main_frame)
frame_img.pack(side=tk.TOP)

frame_ctrl = tk.Frame(main_frame)
frame_ctrl.pack(side=tk.BOTTOM, fill=tk.X)

# Empty image placeholder
img_label = tk.Label(frame_img, text="Load an image/video to begin")
img_label.pack()

stats_var = tk.StringVar(value="No image loaded")

stats_label = tk.Label(
    frame_ctrl,
    textvariable=stats_var,
    justify="left",
    anchor="w",
    font=("Courier New", 10)
)

stats_legend_frame = tk.Frame(frame_ctrl)
stats_legend_frame.pack(fill="x", pady=(10, 0))

stats_label = tk.Label(
    stats_legend_frame,
    textvariable=stats_var,
    justify="left",
    anchor="w",
    font=("Courier New", 10)
)
stats_label.pack(side=tk.LEFT, fill="x", expand=True, ipady=5)

# Legend
legend_frame = tk.Frame(stats_legend_frame, padx=10)
legend_frame.pack(side=tk.RIGHT)

tk.Label(legend_frame, text="Legend", font=("Courier New", 10, "bold")).pack(anchor="w")

legend_row(legend_frame, "blue",   "Small")
legend_row(legend_frame, "yellow", "Medium")
legend_row(legend_frame, "red",    "Large")

# Sliders
param2_var = tk.IntVar()
minr_var = tk.IntVar()
maxr_var = tk.IntVar()

param2_scale = tk.Scale(
    frame_ctrl,
    label="param2",
    from_=10,
    to=120,
    orient=tk.HORIZONTAL,
    variable=param2_var,
    length=500,
    state="disabled",
    command=update_image
)

minr_scale = tk.Scale(
    frame_ctrl,
    label="minRadius",
    from_=1,
    to=1,
    orient=tk.HORIZONTAL,
    variable=minr_var,
    length=600,
    state="disabled",
    command=update_image
)

maxr_scale = tk.Scale(
    frame_ctrl,
    label="maxRadius",
    from_=2,
    to=2,
    orient=tk.HORIZONTAL,
    variable=maxr_var,
    length=500,
    state="disabled",
    command=update_image
)

param2_scale.pack(fill="x")
minr_scale.pack(fill="x")
maxr_scale.pack(fill="x")

# Buttons
button_frame = tk.Frame(frame_ctrl)
button_frame.pack(fill="x", pady=10)

tk.Button(
    button_frame,
    text="Auto Tune",
    command=auto_tune_button,
    height=2
).pack(side=tk.LEFT, expand=True, fill="x", padx=5)


tk.Button(
    button_frame,
    text="Load Image",
    command=load_image,
    height=2
).pack(side=tk.LEFT, expand=True, fill="x", padx=5)

tk.Button(
    button_frame,
    text="Quick Save",
    command=quick_save,
    height=2
).pack(side=tk.LEFT, expand=True, fill="x", padx=5)

tk.Button(
    button_frame,
    text="Save As...",
    command=save_as,
    height=2
).pack(side=tk.LEFT, expand=True, fill="x", padx=5)

tk.Button(
    button_frame,
    text="Load Video",
    command=load_video,
    height=2
).pack(side=tk.LEFT, expand=True, fill="x", padx=5)

# Update idle tasks to calculate widget sizes
root.update_idletasks()

# Start the Tkinter main loop
root.mainloop()


