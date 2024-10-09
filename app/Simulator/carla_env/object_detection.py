import argparse
import os
import time
from PIL import Image
import cv2
import numpy as np
from sahi.predict import get_sliced_prediction, get_prediction
from sahi.models.yolov8 import Yolov8DetectionModel
from yolox.tracker.byte_tracker import BYTETracker
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from .enums import CAM_DIRECTION

# from ultralytics.trackers.byte_tracker import BYTETracker


# Define the class IDs we only want to detect (person and vehicle)
allowed_class_ids = [0, 1, 2, 3, 5, 7]  
# 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
vehicle_class_ids = [1, 2, 3, 5, 7]
    
# Initialize YOLOv8 model
model_path = 'models/yolov8m.pt'
model = Yolov8DetectionModel(model_path=model_path, device='cuda', confidence_threshold = 0.3)  # 0.3
# model = YOLO(model_path)

args_state_NORTH = argparse.Namespace(track_thresh=0.3, match_thresh=0.6, track_buffer=50, mot20=False)  # Set this to True if you're using the MOT20 dataset
args_state_EAST = argparse.Namespace(track_thresh=0.3, match_thresh=0.6, track_buffer=50, mot20=False)  # Set this to True if you're using the MOT20 dataset
args_state_SOUTH = argparse.Namespace(track_thresh=0.3, match_thresh=0.6, track_buffer=50, mot20=False)  # Set this to True if you're using the MOT20 dataset
args_state_WEST = argparse.Namespace(track_thresh=0.3, match_thresh=0.6, track_buffer=50, mot20=False)  # Set this to True if you're using the MOT20 dataset
args_reward_NORTH = argparse.Namespace(track_thresh=0.3, match_thresh=0.6, track_buffer=50, mot20=False)  # Set this to True if you're using the MOT20 dataset
args_reward_EAST = argparse.Namespace(track_thresh=0.3, match_thresh=0.6, track_buffer=50, mot20=False)  # Set this to True if you're using the MOT20 dataset
args_reward_SOUTH = argparse.Namespace(track_thresh=0.3, match_thresh=0.6, track_buffer=50, mot20=False)  # Set this to True if you're using the MOT20 dataset
args_reward_WEST = argparse.Namespace(track_thresh=0.3, match_thresh=0.6, track_buffer=50, mot20=False)  # Set this to True if you're using the MOT20 dataset

# args = argparse.Namespace(track_buffer=30)
byte_tracker_state_NORTH = BYTETracker(args_state_NORTH, frame_rate=24)
byte_tracker_state_EAST = BYTETracker(args_state_EAST, frame_rate=24)
byte_tracker_state_SOUTH = BYTETracker(args_state_SOUTH, frame_rate=24)
byte_tracker_state_WEST = BYTETracker(args_state_WEST, frame_rate=24)

byte_tracker_reward_NORTH = BYTETracker(args_reward_NORTH, frame_rate=24)
byte_tracker_reward_EAST = BYTETracker(args_reward_EAST, frame_rate=24)
byte_tracker_reward_SOUTH = BYTETracker(args_reward_SOUTH, frame_rate=24)
byte_tracker_reward_WEST = BYTETracker(args_reward_WEST, frame_rate=24)

byte_trackers_state = [byte_tracker_state_NORTH, byte_tracker_state_EAST, byte_tracker_state_SOUTH, byte_tracker_state_WEST]
byte_trackers_reward = [byte_tracker_reward_NORTH, byte_tracker_reward_EAST, byte_tracker_reward_SOUTH, byte_tracker_reward_WEST]

# # Initialize ByteTrack
# byte_tracker = BYTETracker(args)

# Frame interval for processing
# frame_interval = 15

def count_vehicles(filtered_detections):
    # Initialize a counter for the current image's "car" objects
    vehicle_count_current_image = 0
    person_count_current_image = 0
        
    # Loop through all detected objects in the current image
    # Check if the object is a vehicle, increment the vehicle count 
        
    vehicle_count_current_image = sum(1 for obj in filtered_detections if obj[5] in vehicle_class_ids) 
    # Check if the object is a person, increment the person count for this image
    # person_count_current_image = sum(1 for obj in filtered_detections if obj[5] == 0) 
    
    # for obj in detections:
    #     if obj.category.id in vehicle_class_ids:  
    #         vehicle_count_current_image += 1 
    #     elif obj.category.id == 0:  
    #         person_count_current_image += 1 
    
    print(f"[Class Counting] {vehicle_count_current_image} vehicles")
    
    return vehicle_count_current_image


def preprocess_frame(cam_direction: CAM_DIRECTION, image_array: np.array):        
    flag = True
    # Convert the screenshot to a NumPy array (RGB format)
    # image_array = np.array(frame)

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
    # Resize the image to enhance the speed
    new_width = 600  # 1280
    ratio = new_width / image.size[0]
    image = image.resize([int(ratio * image.size[0]),int(ratio * image.size[1])], Image.Resampling.LANCZOS)

    # convert to grayscale to reduce input complexity
    # image = image.convert('L')  # 'L' mode is for grayscale 

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

    # if (cam_direction == CAM_DIRECTION.NORTH):
    #     roi_polygon_points = np.array([53, 95,244, 82,595, 432,595, 718,272, 798,52, 797])  # ROI coordinates from original image frame
    # elif (cam_direction == CAM_DIRECTION.EAST):
    #     roi_polygon_points = np.array([53, 95,244, 82,595, 432,595, 718,272, 798,52, 797])  # ROI coordinates from original image frame
    # elif (cam_direction == CAM_DIRECTION.SOUTH):
    #     roi_polygon_points = np.array([53, 95,244, 82,595, 432,595, 718,272, 798,52, 797])  # ROI coordinates from original image frame
    # elif (cam_direction == CAM_DIRECTION.WEST):
    #     roi_polygon_points = np.array([53, 95,244, 82,595, 432,595, 718,272, 798,52, 797])  # ROI coordinates from original image frame
    
    if cam_direction == CAM_DIRECTION.NORTH:
        roi_polygon_points = np.array([(228, 130),(346, 127),(600, 603),(599, 797),(0, 799),(2, 419)])
    elif cam_direction == CAM_DIRECTION.EAST:
        roi_polygon_points = np.array([(228, 130),(346, 127),(600, 603),(599, 797),(0, 799),(2, 419)])
    elif cam_direction == CAM_DIRECTION.SOUTH:
        roi_polygon_points = np.array([(228, 130),(346, 127),(600, 603),(599, 797),(0, 799),(2, 419)])
    elif cam_direction == CAM_DIRECTION.WEST:
        roi_polygon_points = np.array([(228, 130),(346, 127),(600, 603),(599, 797),(0, 799),(2, 419)])

    # roi_polygon_points = np.array([(71, 156),(247, 116),(597, 504),(596, 682),(251, 793),(101, 792)])  # ROI coordinates from original image frame
    roi_polygon_points = (roi_polygon_points * ratio).astype(np.int32)  # converted coordinates by applying resizing ratio 


    # 4. Return the preprocessed image
    # print(roi_polygon_points.shape)
    # print(roi_polygon_points)

    return image_array, roi_polygon_points   # (80, 80, 1) is returned!


def get_carla_frame(cam_direction: CAM_DIRECTION):
    ### ###
    image = None
    image_path = None

    # if cam_direction == CAM_DIRECTION.NORTH:
    #     image_path = r"traffic videos\carla_video_01\1.png"
    # elif cam_direction == CAM_DIRECTION.EAST:
    #     image_path = r"traffic videos\carla_video_01\2.png"
    # elif cam_direction == CAM_DIRECTION.SOUTH:
    #     image_path = r"traffic videos\carla_video_01\3.png"
    # elif cam_direction == CAM_DIRECTION.WEST:
    #     image_path = r"traffic videos\carla_video_01\4.png"



    ### ###
    # image = Image.open(image_path)
    
    #image_array = np.array(image)
    
    return image

def get_carla_status_by_image(cam_direction: CAM_DIRECTION, frame_interval: int):
    
    
    
    return

def get_carla_status_by_numbers(image_array: np.array, cam_direction: CAM_DIRECTION, frame_interval: int):
    stats = []
    
    frame = get_carla_frame(cam_direction)
    
    frame, roi_polygon_points = preprocess_frame(cam_direction, frame)
    roi_polygon = Polygon(roi_polygon_points)

    # Draw the ROI on the frame (optional, for visualization)
    cv2.polylines(frame, [roi_polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Perform YOLOv8 + SAHI detection on each frame
    result = get_sliced_prediction(
        image=frame, # img_path
        detection_model = model,
        slice_height=650, # 770  # 256
        slice_width=650,  # 770  # 256
        overlap_height_ratio=0.25, # 0.25
        overlap_width_ratio=0.25,  # 0.25
        postprocess_type="GREEDYNMM",  # Use NMS for handling overlapping boxes  # originally: GREEDYNMM
        postprocess_match_metric="IOS",  # Use IoU as the matching metric  # originally: IOS
        postprocess_match_threshold=0.3,  # Lower threshold to handle nested boxes  # originally: 0.5
        postprocess_class_agnostic=False
    )
    # result = get_prediction(frame, model)
    # result = model(frame)
    detections0 = result.to_coco_annotations()
    
    # Collect all detected objects
    detections1 = result.object_prediction_list
      
    # print(detections0)
    # print(detections1)
    
    # print(f"<< frame NO. {frame_count} >>")
    print(f"<< CAMERA {cam_direction} Object Detection >>")

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
            
    stats.append(count_vehicles(detection_results))

    detection_results = np.array(detection_results)  # Convert list to NumPy array

    online_targets = []
    
    # Check the shape of detection_results
    if detection_results.size > 0:
        # Apply ByteTrack for tracking only within ROI
        online_targets = byte_trackers_state[cam_direction.value].update(  #         online_targets = byte_trackers[cam_direction].update(
            output_results=detection_results, 
            img_info=[frame.shape[0], frame.shape[1]], 
            img_size=[frame.shape[0], frame.shape[1]]
            )
        # else:
        #     print("Detection results do not have the expected shape.")
        #     online_targets = []  # Handle unexpected shape        

    print(online_targets)
     
    # Loop through tracked objects and draw bounding boxes with track IDs
    for track in online_targets:
        track_id = track.track_id
        score = track.score
        bbox = [int(i) for i in track.tlbr] # (top, left, width, height)  # Convert to int for drawing
        # bbox_center = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)       
        bbox_center_x, bbox_center_y  = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        lasted_frames = (track.end_frame-track.start_frame)*frame_interval+1

        print(f"[track ID {track_id}] location: {bbox} started: frame {(track.start_frame-1)*frame_interval}, duration: lasted {lasted_frames} frames")
        stats.extend([bbox_center_x, bbox_center_y, lasted_frames])
        
        if (len(stats)==1+90):
            break
        
        
        # Draw bounding box and track ID if inside ROI
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), 1) # {score:.2f}
        
    for i in range(91-len(stats)):
        stats.extend([0])
        
    print(f"len(stats): {len(stats)}")
            
    print()

    # # # # Optionally display the result
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     # out.release()
    #     cv2.destroyAllWindows()
    

    return stats  # returns list of length 91 (total 91 numbers, with '0' paddings)
                  # stats = [vehicle_count, bbox_center_x, bbox_center_y, lasted_frames, 
                  #                         bbox_center_x, bbox_center_y, lasted_frames, 
                  #                         ..., 
                  #                         bbox_center_x, bbox_center_y, lasted_frames, 
                  #                                                              0, ..., 0]
    

# # Release everything if the job is done
# out.release()
# cv2.destroyAllWindows()



def caculate_reward_from_carla(image_array: np.array, cam_direction: CAM_DIRECTION, frame_interval: int):
    ## 1. return the total number of cars in 4 ROIs
    ## 2. return the total lasting times(frames) of cars in 4 ROIs
    
    reward = []
    
    # frame = get_carla_frame(cam_direction)
    
    # print("\t\t>> preprocess_frame...", end="")
    frame, roi_polygon_points = preprocess_frame(cam_direction, image_array)
    # print("ok")

    ###debugging###
    # print(frame.shape)
    # cv2.imwrite("_out/debugging_frame.png", frame)
    ###############

    # print("\t\t>> Polygon...", end="")
    roi_polygon = Polygon(roi_polygon_points)
    # print("ok")

    # Draw the ROI on the frame (optional, for visualization)
    # print("\t\t>> cv2.polylines...", end="")
    cv2.polylines(frame, [roi_polygon_points], isClosed=True, color=(255, 0, 0), thickness  =2)
    # print("ok")
    

    # Perform YOLOv8 + SAHI detection on each frame
    # print("\t\t>> get_sliced_prediction...", end="")
    result = get_sliced_prediction(
        image=frame, # img_path
        detection_model = model,
        slice_height=600, # 770 # 256
        slice_width=600,  # 770 # 256
        overlap_height_ratio=0.25, # 0.25
        overlap_width_ratio=0.25,  # 0.25
        postprocess_type="GREEDYNMM",  # Use NMS for handling overlapping boxes  # originally: GREEDYNMM
        postprocess_match_metric="IOS",  # Use IoU as the matching metric  # originally: IOS
        postprocess_match_threshold=0.3,  # Lower threshold to handle nested boxes  # originally: 0.5
        postprocess_class_agnostic=False
    )
    # print("ok")
    # result = get_prediction(frame, model)
    # result = model(frame)

    # print("\t\t>> result.to_coco_annotations...", end="")
    detections0 = result.to_coco_annotations()
    # print("ok")
    
    # Collect all detected objects
    detections1 = result.object_prediction_list
      
    # print(detections0)
    # print(detections1)
    
    # print(f"<< frame NO. {frame_count} >>")
    
    
    # print(f"<< CAMERA {cam_direction} Object Detection >>")          ################  <<=====   ##############

    # Filter detections based on whether they are inside the ROI
    detection_results = []
    print("\t\t>> for d in detections0:...", end="")
    for d in detections0:
        # Extract bounding box coordinates (xmin, ymin, width, height)
        xmin, ymin, width, height = d['bbox']
        xmax, ymax = xmin + width, ymin + height

        # Check if the center of the bounding box is inside the ROI polygon
        bbox_center = Point((xmin + xmax) / 2, (ymin + ymax) / 2)
        if roi_polygon.contains(bbox_center) and d['category_id'] in allowed_class_ids:
            detection_results.append([xmin, ymin, xmax, ymax, d['score'], d['category_id']])
            
    # print("ok")

    # print("\t\t>> count_vehicles...", end="")
    reward.append(count_vehicles(detection_results))
    # print("ok")

    detection_results = np.array(detection_results)  # Convert list to NumPy array

    online_targets = []
    
    # Check the shape of detection_results
    # print("\t\t>> update...", end="")
    ## error occur here ##
    if detection_results.size > 0:
        # Apply ByteTrack for tracking only within ROI
        online_targets = byte_trackers_reward[cam_direction.value].update(  #         online_targets = byte_trackers[cam_direction].update(
            output_results=detection_results, 
            img_info=[frame.shape[0], frame.shape[1]], 
            img_size=[frame.shape[0], frame.shape[1]]
            )
        # else:
        #     print("Detection results do not have the expected shape.")
        #     online_targets = []  # Handle unexpected shape        
    # print("ok")

    i = 0
    sum_lasted_frame = 0
    
    # Loop through tracked objects and draw bounding boxes with track IDs
    # print("\t\t>> for track in online_targets:...", end="")
    for track in online_targets:
        track_id = track.track_id
        score = track.score
        bbox = [int(i) for i in track.tlbr] # (top, left, width, height)  # Convert to int for drawing
        # bbox_center = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)       
        bbox_center_x, bbox_center_y  = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        lasted_frames = (track.end_frame-track.start_frame)*frame_interval+1

        # print(f"[track ID {track_id}] location: {bbox} started: frame {(track.start_frame-1)*frame_interval}, duration: lasted {lasted_frames} frames")     ################  <<=====   ##############
        sum_lasted_frame += lasted_frames
        i += 1
        
        if (i==30):
            break
        
        # Draw bounding box and track ID if inside ROI
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), 1) # {score:.2f}
    
    # print("ok")
    reward.append(sum_lasted_frame)
            
    print()

    if cam_direction == CAM_DIRECTION.NORTH:
        # # Optionally display the result
        cv2.imshow('Frame', frame)                        ################  <<=====   ##############
        
        # # image = Image.fromarray(frame)
        # # # image.save(f"{cam_direction}.png")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #out.release()
            cv2.destroyAllWindows()

    # total_average_waiting_time = sum_total_waiting_time / sum_waiting_vehicle_count if sum_waiting_vehicle_count > 0 else 0
    # reward = -1 * total_average_waiting_time   # 모든 대기차량의 평균 대기시간

    return reward  # returns list of length 2
                   # reward = [vehicle_count, sum_lasted_frame]
    


# # Release everything if the job is done
# out.release()
# cv2.destroyAllWindows()





 

