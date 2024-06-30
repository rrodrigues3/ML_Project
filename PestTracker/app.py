import gradio as gr
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import io

def render_result(image, boxes):
    """Render the results with bounding boxes on the image."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])

        # Definindo a cor e o texto com base na classe
        if cls == 0:  # Supondo que a classe 0 seja a mosca
            color = (0, 255, 0)  # Verde para moscas
            label = f'Mosca: {conf:.2f}'
        else:
            color = (0, 0, 255)  # Vermelho para outros objetos
            label = f'Outro: {conf:.2f}'

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Put confidence score
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def yoloV8_func(image):
    """This function performs YOLOv8 object detection on the given image."""
    # Define default parameters
    image_size = 640
    conf_threshold = 0.25
    iou_threshold = 0.45

    # Load the YOLOv8 model (using a default model for testing)
    model_path = "best.pt"  # You can change this to 'best.pt' if it's a valid model
    model = YOLO(model_path)

    # Perform object detection on the input image using the YOLOv8 model
    results = model.predict(image,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            imgsz=image_size)

    # Print the detected objects' information (class, coordinates, and probability)
    box = results[0].boxes
    print("Object type:", box.cls)
    print("Coordinates:", box.xyxy)
    print("Probability:", box.conf)

    # Convert image to numpy array if necessary
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Render the output image with bounding boxes around detected objects
    rendered_image = render_result(image, box)
    
    # Count the number of detected objects
    num_objects = len(box)
    num_flies = sum(1 for b in box if int(b.cls[0]) == 0)
    num_others = num_objects - num_flies

    # Convert the rendered image to PIL format for display in Gradio
    rendered_image_pil = Image.fromarray(cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB))
    
    detection_info = (f'Number of objects detected: {num_objects}\n'
                      f'Number of other objects detected: {num_others}')
    
    return rendered_image_pil, detection_info

# Define the Gradio interface
inputs = gr.Image(type="filepath", label="Input Image")

outputs = [gr.Image(type="pil", label="Output Image"), gr.Textbox(label="Detection Info")]

title = "YOLOv8 101: Deteção de mosca"

examples = [['one.jpg'],
            ['two.jpg'],
            ['three.jpg']]

yolo_app = gr.Interface(
    fn=yoloV8_func,
    inputs=inputs,
    outputs=outputs,
    title=title,
    examples=examples,
    cache_examples=True,
)

# Launch the Gradio interface in debug mode
yolo_app.launch(debug=True)
