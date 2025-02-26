import gradio as gr
import torch
from ultralyticsplus import YOLO, render_result


torch.hub.download_url_to_file(
    'https://www.vidarural.pt/wp-content/uploads/sites/5/2015/11/Mosca_Figura-3.jpg', 'one.jpg')
torch.hub.download_url_to_file(
    'https://iplantprotect.pt/wp/wp-content/uploads/2022/04/Armadilha_cropped-1024x678.jpg', 'two.jpg')
torch.hub.download_url_to_file(
    'https://www.biosani.com/uploads/produtos_galeria/rebell-amarillo-1.jpg', 'three.jpg')


def yoloV8_func(image: gr.inputs.Image = None,
                image_size: gr.inputs.Slider = 640,
                conf_threshold: gr.inputs.Slider = 0.4,
                iou_threshold: gr.inputs.Slider = 0.50):
    """This function performs YOLOv8 object detection on the given image.

    Args:
        image (gr.inputs.Image, optional): Input image to detect objects on. Defaults to None.
        image_size (gr.inputs.Slider, optional): Desired image size for the model. Defaults to 640.
        conf_threshold (gr.inputs.Slider, optional): Confidence threshold for object detection. Defaults to 0.4.
        iou_threshold (gr.inputs.Slider, optional): Intersection over Union threshold for object detection. Defaults to 0.50.
    """
    # Load the YOLOv8 model from the 'best.pt' checkpoint
    model_path = "best.pt"
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

    # Render the output image with bounding boxes around detected objects
    render = render_result(model=model, image=image, result=results[0])
    return render


inputs = [
    gr.inputs.Image(type="filepath", label="Input Image"),
    gr.inputs.Slider(minimum=320, maximum=1280, default=640,
                     step=32, label="Image Size"),
    gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.25,
                     step=0.05, label="Confidence Threshold"),
    gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.45,
                     step=0.05, label="IOU Threshold"),
]


outputs = gr.outputs.Image(type="filepath", label="Output Image")

title = "YOLOv8 101: Custom Object Detection on Construction Workers"


examples = [['one.jpg', 640, 0.5, 0.7],
            ['two.jpg', 800, 0.5, 0.6],
            ['three.jpg', 900, 0.5, 0.8]]

yolo_app = gr.Interface(
    fn=yoloV8_func,
    inputs=inputs,
    outputs=outputs,
    title=title,
    examples=examples,
    cache_examples=True,
)

# Launch the Gradio interface in debug mode with queue enabled
yolo_app.launch(debug=True, enable_queue=True)


__________________________

import gradio as gr
import torch
from ultralytics import YOLO
import cv2
import numpy as np

torch.hub.download_url_to_file(
    'https://www.vidarural.pt/wp-content/uploads/sites/5/2015/11/Mosca_Figura-3.jpg', 'one.jpg')
torch.hub.download_url_to_file(
    'https://iplantprotect.pt/wp/wp-content/uploads/2022/04/Armadilha_cropped-1024x678.jpg', 'two.jpg')
torch.hub.download_url_to_file(
    'https://www.biosani.com/uploads/produtos_galeria/rebell-amarillo-1.jpg', 'three.jpg')

def render_result(image, boxes):
    """Render the results with bounding boxes on the image."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = box.cls[0]

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put confidence score
        text = f'{cls}: {conf:.2f}'
        cv2.putText(image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def yoloV8_func(image, image_size, conf_threshold, iou_threshold):
    """This function performs YOLOv8 object detection on the given image.

    Args:
        image: Input image to detect objects on.
        image_size: Desired image size for the model.
        conf_threshold: Confidence threshold for object detection.
        iou_threshold: Intersection over Union threshold for object detection.
    """
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
    output_path = 'output.jpg'
    cv2.imwrite(output_path, rendered_image)
    return output_path

inputs = [
    gr.Image(type="filepath", label="Input Image"),
    gr.Slider(minimum=320, maximum=1280, value=640, step=32, label="Image Size"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.45, step=0.05, label="IOU Threshold"),
]

outputs = gr.Image(type="filepath", label="Output Image")

title = "YOLOv8 101: Deteção de mosca"

examples = [['one.jpg', 640, 0.5, 0.7],
            ['two.jpg', 800, 0.5, 0.6],
            ['three.jpg', 900, 0.5, 0.8]]

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
