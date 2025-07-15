# Group Members 

Maryam Basharat

Aimen Zahra

Iqra Ramzan

# YOLO-Object-Detection-

This project implements an object detection system using YOLOv8 (You Only Look Once) by Ultralytics. It detects and classifies objects in images and videos with high speed and accuracy. The entire workflow is built and tested using Visual Studio Code (VS Code) for development.

# Features

Object detection using YOLOv8 (pre-trained or custom trained).

Image, video, and webcam input support.

VS Code environment with Python support.

Easy to modify for different datasets and detection tasks.

# Libraries Used

Ultralytics YOLO: For loading and running the YOLOv8 pre-trained model and object tracking.
OpenCV: For video processing and image manipulation.
Pillow (PIL): For converting OpenCV images to a format compatible with Tkinter GUI.
Tkinter: To build the graphical user interface.

## Requirements

- Python 3.8+
- Ultralytics YOLO
- OpenCV (for video/camera)

Install:

```bash
pip install ultralytics opencv-python

# Run Detection

python yolov8_object_detection.py

Inside the script, choose between image, video, or webcam input.

By Default, YOLOv8 Can Detect These 80 Classes (COCO Dataset):

# People

person

# Vehicles

bicycle

car

motorcycle

airplane

bus

train

truck

boat

# Animals

bird

cat

dog

horse

sheep

cow

elephant

bear

zebra

giraffe

# Food / Furniture

backpack

umbrella

handbag

tie

suitcase

chair

couch

potted plant

bed

dining table

toilet

# Electronics

TV

laptop

mouse

remote

keyboard

cell phone

microwave

oven

toaster

sink

refrigerator

# Sports / Toys

book

clock

vase

scissors

teddy bear

hair drier

toothbrush

sports ball

kite

baseball bat

skateboard

surfboard

tennis racket

## Tools / Objects

knife

fork

spoon

bottle

wine glass

cup

bowl

sandwich

apple

orange

banana

hot dog

pizza

donut

cake

## Custom Detection
YOLOv8 can also detect custom objects like:

License plates

Weapons

Trash/waste

Medical tools

Custom logos

Industrial parts

…but you'll need to train it on a dataset labeled with those objects (using your own data.yaml and images/labels).

# About YOLOv8 Model

YOLO (You Only Look Once) is a popular real-time object detection architecture known for its speed and accuracy.
YOLOv8 is the latest version from Ultralytics with improvements in detection quality, speed, and usability.
The model used here is yolov8s.pt (small version) which balances accuracy and speed, suitable for real-time applications.
The model supports both object detection and tracking.
