import cv2
from tkinter import *
from tkinter import filedialog
from threading import Thread
from ultralytics import YOLO
from PIL import Image, ImageTk
import os
from datetime import datetime

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Create output folder
output_dir = "detections"
os.makedirs(output_dir, exist_ok=True)

# Initialize main window
window = Tk()
window.title("YOLO Object Detection")
window.geometry("600x550")
window.configure(bg="lightblue")

video_path = ""
stop_flag = False

# Select video file
def select_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    label_status.config(text=f"Selected: {os.path.basename(video_path) if video_path else 'None'}")

# Select image and detect objects
def select_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not image_path:
        return

    results = model(image_path)
    for r in results:
        annotated = r.plot()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"image_{timestamp}.jpg")
        cv2.imwrite(save_path, annotated)

        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated)
        imgtk = ImageTk.PhotoImage(image=img)

        img_label = Label(window, image=imgtk)
        img_label.image = imgtk
        img_label.pack()

        label_status.config(text=f"Saved result to: {save_path}")

# Detect objects in video
def detect_video(source=0, is_webcam=False):
    global stop_flag
    stop_flag = False
    class_filter = entry_class.get().strip().lower()

    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"video_{timestamp}.avi")
    out = None

    while cap.isOpened() and not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for r in results:
            boxes = r.boxes
            names = r.names
            if class_filter:
                keep = []
                for i, cls_id in enumerate(boxes.cls):
                    class_name = names[int(cls_id)]
                    if class_name.lower() == class_filter:
                        keep.append(i)
                if keep:
                    r.boxes = boxes[keep]
                    annotated = r.plot()
                else:
                    annotated = frame
            else:
                annotated = r.plot()

        if out is None:
            h, w = annotated.shape[:2]
            out = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))

        out.write(annotated)
        cv2.imshow("YOLOv8 Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
        label_status.config(text=f"Saved video to: {save_path}")
    cv2.destroyAllWindows()

# Start webcam detection
def start_webcam():
    Thread(target=detect_video, args=(0, True)).start()

# Start video file detection
def start_video_detection():
    if not video_path:
        label_status.config(text="No video selected.")
        return
    Thread(target=detect_video, args=(video_path, False)).start()

# Stop detection
def stop_detection():
    global stop_flag
    stop_flag = True

# GUI widgets
Label(window, text="YOLO Object Detection", font=("Arial", 18, "bold"), fg="darkred", bg="lightblue").pack(pady=10)

label_status = Label(window, text="No source selected", bg="lightblue")
label_status.pack(pady=5)

Button(window, text="Select Video", bg="blue", fg="white", command=select_video).pack(pady=5)
Button(window, text="Select Image", bg="blue", fg="white", command=select_image).pack(pady=5)
Button(window, text="Start Webcam", bg="purple", fg="white", command=start_webcam).pack(pady=5)

Label(window, text="Enter Class to Detect (optional):", bg="lightblue").pack()
entry_class = Entry(window)
entry_class.pack(pady=5)

Button(window, text="Start Detection", bg="green", fg="white", command=start_video_detection).pack(pady=5)
Button(window, text="Stop Detection", bg="red", fg="white", command=stop_detection).pack(pady=5)

# Main GUI loop
window.mainloop()
