import argparse
import time
from PIL import Image
import cv2
import numpy as np
from sahi.predict import get_sliced_prediction, get_prediction
from sahi.models.yolov8 import Yolov8DetectionModel
from yolox.tracker.byte_tracker import BYTETracker
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
# from ultralytics.trackers.byte_tracker import BYTETracker

# Define the class IDs we only want to detect (person and vehicle)
allowed_class_ids = [0, 1, 2, 3, 5, 7]  
# 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
vehicle_class_ids = [1, 2, 3, 5, 7]


def count_vehicles_and_people(filtered_detections):
    # Initialize a counter for the current image's "car" objects
    vehicle_count_current_image = 0
    person_count_current_image = 0
        
    # Loop through all detected objects in the current image
    # Check if the object is a vehicle, increment the vehicle count 
        
    vehicle_count_current_image = sum(1 for obj in filtered_detections if obj[5] in vehicle_class_ids) 
    # Check if the object is a person, increment the person count for this image
    person_count_current_image = sum(1 for obj in filtered_detections if obj[5] == 0) 
    
    # for obj in detections:
    #     if obj.category.id in vehicle_class_ids:  
    #         vehicle_count_current_image += 1 
    #     elif obj.category.id == 0:  
    #         person_count_current_image += 1 
    
    print(f"[Class Counting] {vehicle_count_current_image} vehicles, {person_count_current_image} people")



def preprocess_frame(frame):        
    flag = True
    
    # Convert the screenshot to a NumPy array (RGB format)
    image_array = np.array(frame)
    
    # if (flag == True):
    #     image = Image.fromarray(image_array)
    #     image.save("sumo_screenshot_before_before.png")

    # The `mss` screenshot contains 4 channels (RGBA), so we can discard the alpha channel
    image_array = image_array[..., :3]  # Keep only the RGB channels

    # Convert the image to a PIL image to use its transformation functions
    image = Image.fromarray(image_array)

    # if (flag == True):
    #     # image = Image.fromarray(image_array)
    #     image.save("sumo_screenshot_before.png")

    # 3. Preprocess the image
    # Resize the image to match the input shape of the CNN
    new_width = 1280
    ratio = new_width / image.size[0]
    image = image.resize([int(ratio * image.size[0]),int(ratio * image.size[1])], Image.Resampling.LANCZOS)

    # convert to grayscale to reduce input complexity
    # image = image.convert('L')  # 'L' mode is for grayscale 
    
    
    # if (flag == True):
    #     # image = Image.fromarray(image_array)
    #     image.save("sumo_screenshot_after.png")
    # flag += False

    # Convert the image back to a NumPy array
    image_array = np.array(image)

    # # Normalize pixel values to be between 0 and 1
    # image_array = image_array / 255.0

    # # Add an extra dimension to match the input shape of the CNN
    # # (80, 80, 1) if grayscale, or (80, 80, 3) for RGB images
    # image_array = np.expand_dims(image_array, axis=-1)

    # Define your polygonal ROI (Region of Interest)
    # Example: polygon with 4 points (adjust this to match your desired ROI) 
    # roi_polygon_points = np.array([[723, 200], [1062,200], [1450, 675], [684, 675], [682, 413]])
    # 
    roi_polygon_points = np.array([(212, 205),(393, 135),(843, 474),(524, 656)]) 
    roi_polygon_points = (roi_polygon_points * ratio).astype(np.int32)


    # 4. Return the preprocessed image
    return image_array, roi_polygon_points   # (80, 80, 1) is returned!

    
# Initialize YOLOv8 model
model_path = 'models/yolov8s.pt'
model = Yolov8DetectionModel(model_path=model_path, device='cpu', confidence_threshold = 0.3)
# model = YOLO(model_path)

args = argparse.Namespace(
    track_thresh=0.3,
    match_thresh=0.6,
    track_buffer=50,
    mot20=False  # Set this to True if you're using the MOT20 dataset
)

# args = argparse.Namespace(track_buffer=30)
byte_tracker = BYTETracker(args, frame_rate=24)

# # Initialize ByteTrack
# byte_tracker = BYTETracker(args)

# Open video file or capture from webcam
video_path = 'traffic videos/carla_video_01.mp4'
cap = cv2.VideoCapture(video_path)

# Define codec and create VideoWriter object for saving the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('traffic videos/carla_video_01_output.mp4', fourcc, 24.0, (int(cap.get(3)), int(cap.get(4))))

# Frame interval for processing
frame_interval = 10
frame_count = 0  # Initialize frame counter

# Dictionary to store entry times for tracking
entry_times = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    

    # Process only every 'frame_interval' frames
    if frame_count % frame_interval != 0:
        # # Update the tracker with an empty detection result to maintain consistency
        #byte_tracker.update(np.array(frame), img_info=[frame.shape[0], frame.shape[1]], img_size=[frame.shape[0], frame.shape[1]])
        
        out.write(frame)  # Write the original frame to output without tracking
        frame_count += 1  # Increment frame counter

        continue  # Skip to the next frame
    
    frame, roi_polygon_points = preprocess_frame(frame)
    roi_polygon = Polygon(roi_polygon_points)

    # Draw the ROI on the frame (optional, for visualization)
    cv2.polylines(frame, [roi_polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Perform YOLOv8 + SAHI detection on each frame
    result = get_sliced_prediction(
        image=frame, # img_path
        detection_model = model,
        slice_height=770, # 256
        slice_width=770,  # 256
        overlap_height_ratio=0.25, # 0.25
        overlap_width_ratio=0.25,  # 0.25
        postprocess_type="GREEDYNMM",  # Use NMS for handling overlapping boxes  # originally: GREEDYNMM
        postprocess_match_metric="IOS",  # Use IoU as the matching metric  # originally: IOS
        postprocess_match_threshold=0.5,  # Lower threshold to handle nested boxes  # originally: 0.5
        postprocess_class_agnostic=False
    )
    # result = get_prediction(frame, model)
    # result = model(frame)
    detections0 = result.to_coco_annotations()
    
    # Collect all detected objects
    detections1 = result.object_prediction_list
      
    # print(detections0)
    # print(detections1)

      
    # Filter detections based on whether they are inside the ROI
    detection_results = []
    for d in detections0:
        # Extract bounding box coordinates (xmin, ymin, width, height)
        xmin, ymin, width, height = d['bbox']
        xmax, ymax = xmin + width, ymin + height

        # Check if the center of the bounding box is inside the ROI polygon
        bbox_center = Point((xmin + xmax) / 2, (ymin + ymax) / 2)
        if roi_polygon.contains(bbox_center) and d['category_id'] in allowed_class_ids:
            detection_results.append([xmin, ymin, xmax, ymax, d['score'], d['category_id']])
            
    
    count_vehicles_and_people(detection_results)

    detection_results = np.array(detection_results)  # Convert list to NumPy array
    # print(detection_results)

    online_targets = []
    
    # Check the shape of detection_results
    if detection_results.size > 0:
        # print(f"Detection results shape: {np.shape(detection_results)}")
        
        #print(detection_results)
        
        # Ensure the results have the expected shape (N, 5) where N is the number of detections
        # if detection_results.ndim == 2 and detection_results.shape[1] == 5:
        # Prepare img_info and img_size based on the current frame
        
        img_h, img_w = frame.shape[:2]  # Get height and width of the frame
        img_info = [img_h, img_w]  # [height, width]
        img_size = [img_h, img_w]   # This can also be set to some other size if needed

        # Apply ByteTrack for tracking only within ROI
        # online_targets = byte_tracker.update(output_results=detection_results, img_info=img_info, img_size=img_size)
        online_targets = byte_tracker.update(
            output_results=detection_results, 
            img_info=[frame.shape[0], frame.shape[1]], 
            img_size=[frame.shape[0], frame.shape[1]]
            )
        # else:
        #     print("Detection results do not have the expected shape.")
        #     online_targets = []  # Handle unexpected shape        

    print(f"<< frame NO. {frame_count} >>")

    # Loop through tracked objects and draw bounding boxes with track IDs
    for track in online_targets:
        track_id = track.track_id
        score = track.score
        bbox = [int(i) for i in track.tlwh] # (top, left, width, height)  # Convert to int for drawing

        print(f"[track ID {track_id}] location: {bbox} started: frame {(track.start_frame-1)*frame_interval}, duration: lasted {(track.end_frame-track.start_frame)*frame_interval+1} frames")

        # print(f"{track.}")
        # print(f"{track_id} : {bbox}")
        
        # Draw bounding box and track ID if inside ROI
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), 1) # {score:.2f}

        # Count objects and track entry/exit times
        
        # class_id = track.cls  # Assuming the class ID is stored in the track object
        # class_count[class_id] += 1
        
        # Check if this track is entering or exiting the ROI
        bbox_center = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)        

        if track_id not in entry_times:
            entry_times[track_id] = frame_count  #time.time()  # Record entry time
            print(f"[IN] ID{track_id} : {entry_times[track_id]}")            
       
           
    # track_id_list = []    
    # for track in online_targets:
    #     track_id_list.append(track.track_id)
    
    # print(track_id_list)
    # print(entry_times.keys()) 
        
    # for key in entry_times.copy().keys():
    #     if key not in track_id_list:
    #         entry_time = entry_times.pop(key)  # Remove entry time
    #         current_time = frame_count
    #         time_spent = current_time - entry_time
    #         print(f"[OUT] ID{key} : {current_time}, (spent {time_spent})")
            
    print()

            

    # # Print the time spent for objects still in the ROI
    # for track_id in entry_times.keys():
    #     current_time_spent = time.time() - entry_times[track_id]
    #     print(f"Track ID: {track_id} is still in the ROI. Time spent: {current_time_spent:.2f} seconds.")
    
    
    
    # # Print detected object counts
    # print("Detected Object Counts:")
    # for class_id, count in class_count.items():
    #     print(f"Class ID {class_id}: {count}")

    # Write frame with bounding boxes and track IDs to the output video
    out.write(frame)

    frame_count += 1  # Increment frame counter
    
    # Optionally display the result
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


# Release everything if the job is done
cap.release()
out.release()
cv2.destroyAllWindows()
