import cv2
import numpy as np
from ultralytics import YOLO
import os
import pathlib

def is_dynamic_object(class_name):
    # Keywords suggesting dynamic/mobile objects
    dynamic_keywords = [
        'person', 'human', 'man', 'woman', 'child', 'bicycle', 'car', 'motorcycle', 
        'airplane', 'bus', 'train', 'truck', 'boat', 'bird', 'cat', 'dog', 'horse', 
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'vehicle', 
        'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'snowboard', 'ball', 
        'kite', 'bat', 'glove', 'skateboard', 'surfboard', 'racket'
    ]
    # Keywords suggesting stationary objects
    stationary_keywords = [
        'table', 'chair', 'sofa', 'bed', 'desk', 'shelf', 'bookcase', 'cabinet', 
        'building', 'house', 'tree', 'plant', 'lamp', 'sign', 'clock', 'mirror', 
        'sink', 'toilet', 'oven', 'refrigerator', 'microwave', 'dishwasher', 
        'bench', 'fence', 'wall', 'door', 'window', 'rug', 'mat', 'curtain'
    ]
    
    class_name_lower = class_name.lower()
    # Check if the class name contains any dynamic keywords
    is_dynamic = any(keyword in class_name_lower for keyword in dynamic_keywords)
    # Check if the class name contains any stationary keywords
    is_stationary = any(keyword in class_name_lower for keyword in stationary_keywords)
    
    # An object is dynamic if it matches dynamic keywords and does not match stationary ones
    return is_dynamic and not is_stationary

def generate_segmentation_overlay(image_path, output_path=None, model_path='yolov8x-seg.pt', conf_threshold=0.5):
    # If output_path is not provided, generate one based on input image name
    if output_path is None:
        image_name = pathlib.Path(image_path).stem
        output_path = str(pathlib.Path(image_path).parent / f"{image_name}_dynamic_segmentation.jpg")
    
    # Load the YOLOv8 segmentation model (using extra-large variant for better accuracy)
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {str(e)}")
    
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Create a copy of the input image for overlay
    overlay_image = image.copy()
    
    # Perform inference with specified confidence threshold
    try:
        results = model.predict(image, conf=conf_threshold)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
    
    # Flag to track if any masks are applied
    masks_applied = False
    
    # Process each detection
    for result in results:
        if result.masks is None:
            print("No masks detected in this result.")
            continue
            
        for mask, box in zip(result.masks.data, result.boxes):
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            # Print all detected objects for debugging
            print(f"Detected object: {class_name} (ID: {class_id}, Confidence: {confidence:.2f})")
            
            # Only process dynamic objects
            if is_dynamic_object(class_name):
                print(f"Applying mask for dynamic object: {class_name}")
                
                # Extract bounding box coordinates
                bbox = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                
                # Draw bounding box (blue, thickness 2)
                cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Add label with class name and confidence score
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(overlay_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Convert mask to numpy array and resize to image dimensions
                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Create a binary mask
                binary_mask = (mask > 0).astype(np.uint8) * 255
                
                # Generate a random color for the mask
                random_color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # Random RGB
                colored_mask = np.zeros_like(image)
                for channel in range(3):
                    colored_mask[:, :, channel] = binary_mask * random_color[channel]
                
                # Overlay the mask on the image with transparency
                alpha = 0.5  # Transparency factor
                overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_mask, alpha, 0.0)
                masks_applied = True
    
    if not masks_applied:
        print("No dynamic objects were detected to apply masks.")
    
    # Save the overlaid image
    cv2.imwrite(output_path, overlay_image)
    print(f"Segmentation masks overlaid and saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Define paths
    input_image = "Images\Picture7.png"  # Replace with your image path
    output_image = "output_segmentation_Isaac_Sim_Dynamic_AutoDetection.jpg\output_segmentation_Picture_7_AutoDetection.jpg"
    
    # Ensure input image exists
    if not os.path.exists(input_image):
        raise FileNotFoundError(f"Input image not found at {input_image}")
    
    # Run the segmentation with overlay
    generate_segmentation_overlay(input_image, output_image)