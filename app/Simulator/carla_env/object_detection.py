import argparse
import os
import time
from PIL import Image
import cv2
import numpy as np
from sahi.predict import get_sliced_prediction, get_prediction
from sahi.models.yolov8 import Yolov8DetectionModel
from yolox.tracker.basetrack import BaseTrack
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
model = Yolov8DetectionModel(model_path=model_path, device='cuda', confidence_threshold = 0.4)  # 0.3
# model = YOLO(model_path)

args = argparse.Namespace(track_thresh=0.3, match_thresh=0.6, track_buffer=30, mot20=False)  # Set this to True if you're using the MOT20 dataset

# # Initialize ByteTrack
# byte_tracker = BYTETracker(args)

# Frame interval for processing
# frame_interval = 15

class ObjectTracking:
    def __init__(self):
        pass
       
    def start(self):
        # cv2.destroyAllWindows()
        # args = argparse.Namespace(track_buffer=30)
        BaseTrack._count = 0

        # frame_rate은 트래킹 버퍼사이즈랑 관계있기에, carla의 fps(or delta)와 관계없음!!
        self.byte_tracker_reward_NORTH = BYTETracker(args, frame_rate=30)  # carla의 fps(or delta)와 관계없음!!
        self.byte_tracker_reward_EAST = BYTETracker(args, frame_rate=30)
        self.byte_tracker_reward_SOUTH = BYTETracker(args, frame_rate=30)
        self.byte_tracker_reward_WEST = BYTETracker(args, frame_rate=30)

        # self.byte_trackers_state = [self.byte_tracker_state_NORTH, self.byte_tracker_state_EAST, self.byte_tracker_state_SOUTH, self.byte_tracker_state_WEST]
        self.byte_trackers_reward = [self.byte_tracker_reward_NORTH, self.byte_tracker_reward_EAST, self.byte_tracker_reward_SOUTH, self.byte_tracker_reward_WEST]


    def count_vehicles(self, filtered_detections):
        # Initialize a counter for the current image's "car" objects
        vehicle_count = 0
        person_count_current_image = 0
            
        # Loop through all detected objects in the current image
        # Check if the object is a vehicle, increment the vehicle count 
            
        # vehicle_count = sum(1 for obj in filtered_detections if obj[5] in vehicle_class_ids) 
        # person_count = sum(1 for obj in filtered_detections if obj[5] in vehicle_class_ids) 
        # bicycle_count = sum(1 for obj in filtered_detections if obj[5] in vehicle_class_ids) 
        # _count = sum(1 for obj in filtered_detections if obj[5] in vehicle_class_ids) 
        vehicle_count = sum(1 for obj in filtered_detections if obj[5] in vehicle_class_ids) 

        # class_list = {0 : 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        # sums = {}
        # for obj in filtered_detections:
        #     v_type = class_list[obj[5]]
        #     sums[v_type] = sums.get(v_type, 0) + 1

        # person_count = sums.get('person', 0)
        # bicycle_count = sums.get('bicycle', 0)
        # car_count = sums.get('car', 0)
        # motorcycle_count = sums.get('motorcycle', 0)
        # bus_count = sums.get('bus', 0)
        # truck_count = sums.get('truck', 0)

        # vehicle_count = bicycle_count + car_count + motorcycle_count + bus_count + truck_count
        # 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'


        # Check if the object is a person, increment the person count for this image
        # person_count_current_image = sum(1 for obj in filtered_detections if obj[5] == 0) 
        
        # for obj in detections:
        #     if obj.category.id in vehicle_class_ids:  
        #         vehicle_count_current_image += 1 
        #     elif obj.category.id == 0:  
        #         person_count_current_image += 1 
        
        # print(f"[Class Counting] vehicles:{vehicle_count}")
        
        return vehicle_count


    def preprocess_frame(self, cam_direction: CAM_DIRECTION, image_array: np.ndarray):        
        flag = True
        # Convert the screenshot to a NumPy array (RGB format)
        # image_array = np.array(frame)

        # if (flag == True):
        #     image = Image.fromarray(image_array)
        #     image.save("sumo_screenshot_before_before.png")

        # The `mss` screenshot contains 4 channels (RGBA), so we can discard the alpha channel
        # image_array = image_array[..., :3]  # Keep only the RGB channels

        # Convert the image to a PIL image to use its transformation functions
        # image = Image.fromarray(image_array)

        # if (flag == True):
        #     # image = Image.fromarray(image_array)
        #     image.save("sumo_screenshot_before.png")

        # 3. Preprocess the image
        # Resize the image to enhance the speed
        # new_width = 600  # 1280
        # ratio = new_width / image.size[0]
        # image = image.resize([int(ratio * image.size[0]),int(ratio * image.size[1])], Image.Resampling.LANCZOS)

        ratio = 1


        # convert to grayscale to reduce input complexity
        # image = image.convert('L')  # 'L' mode is for grayscale 

        # Convert the image back to a NumPy array
        # image_array = np.array(image)

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
        
        # if cam_direction == CAM_DIRECTION.NORTH:
        #     roi_polygon_points = np.array([(228, 130),(346, 127),(600, 603),(599, 797),(0, 799),(2, 419)])
        # elif cam_direction == CAM_DIRECTION.EAST:
        #     roi_polygon_points = np.array([(228, 130),(346, 127),(600, 603),(599, 797),(0, 799),(2, 419)])
        # elif cam_direction == CAM_DIRECTION.SOUTH:
        #     roi_polygon_points = np.array([(228, 130),(346, 127),(600, 603),(599, 797),(0, 799),(2, 419)])
        # elif cam_direction == CAM_DIRECTION.WEST:
        #     roi_polygon_points = np.array([(228, 130),(346, 127),(600, 603),(599, 797),(0, 799),(2, 419)])

        roi_polygon_points_straight_and_right = np.array([(226, 82),(321, 80),(447, 791),(3, 792),(3, 416)])
        roi_polygon_points_left = np.array([(312, 80),(360, 79),(597, 541),(598, 788),(392, 789)])

        # roi_polygon_points = np.array([(71, 156),(247, 116),(597, 504),(596, 682),(251, 793),(101, 792)])  # ROI coordinates from original image frame
        roi_polygon_points_straight_and_right = (roi_polygon_points_straight_and_right * ratio).astype(np.int32)  # converted coordinates by applying resizing ratio 
        roi_polygon_points_left = (roi_polygon_points_left * ratio).astype(np.int32)  # converted coordinates by applying resizing ratio 

        # 4. Return the preprocessed image
        # print(roi_polygon_points.shape)
        # print(roi_polygon_points)

        return image_array, roi_polygon_points_straight_and_right, roi_polygon_points_left  # (80, 80, 1) is returned!





    def calculate_vehicle_count(self, frame: np.ndarray, cam_direction: CAM_DIRECTION):
        ## 1. return the total number of cars in 4 ROIs
        ## 2. return the total lasting times(frames) of cars in 4 ROIs
                
        # frame = get_carla_frame(cam_direction)
        
        # print("\t\t>> preprocess_frame...", end="")
        # frame, roi_polygon_points_straight_and_right, roi_polygon_points_left = self.preprocess_frame(cam_direction, image_array)
        roi_polygon_points_straight_and_right = np.array([(258, 28),(314, 29),(451, 799),(4, 801),(3, 369)])
        roi_polygon_points_left = np.array([(307, 27),(332, 26),(598, 513),(598, 799),(391, 798)])
        # print("ok")

        ###debugging###
        # print(frame.shape)
        # cv2.imwrite("_out/debugging_frame.png", frame)
        ###############

        # print("\t\t>> Polygon...", end="")
        roi_polygon_straight_and_right = Polygon(roi_polygon_points_straight_and_right)
        roi_polygon_left = Polygon(roi_polygon_points_left)

        # print("ok")

        # Draw the ROI on the frame (optional, for visualization)
        # print("\t\t>> cv2.polylines...", end="")
        cv2.polylines(frame, [roi_polygon_points_straight_and_right], isClosed=True, color=(255, 0, 0), thickness = 2)
        cv2.putText(frame, f'ROI:SR', (226, 82 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 0, 0), 1) # {score:.2f}

        cv2.polylines(frame, [roi_polygon_points_left], isClosed=True, color=(255, 255, 0), thickness=2)
        cv2.putText(frame, f'ROI:L', (312, 80 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 0), 1) # {score:.2f}

        # print("ok")
        

        # Perform YOLOv8 + SAHI detection on each frame
        #print("\t\t>> get_sliced_prediction...", end="")
        result = get_sliced_prediction(
            image=frame, # img_path
            detection_model = model,
            slice_height=650, # 770 # 256
            slice_width=650,  # 770 # 256
            overlap_height_ratio=0.25, # 0.25
            overlap_width_ratio=0.25,  # 0.25
            postprocess_type="GREEDYNMM",  # Use NMS for handling overlapping boxes  # originally: GREEDYNMM
            postprocess_match_metric="IOS",  # Use IoU as the matching metric  # originally: IOS
            postprocess_match_threshold=0.2,  # Lower threshold to handle nested boxes  # originally: 0.5
            postprocess_class_agnostic=False,
            verbose=0
        )
        # time.sleep(0.001)
        # print("ok")
        # result = get_prediction(frame, model)
        # result = model(frame)

        # print("\t\t>> result.to_coco_annotations...", end="")
        detections0 = result.to_coco_annotations()

        vehicle_count_straight_and_right = 0
        vehicle_count_left = 0
        total_vehicle_count = 0

        for d in detections0:
            # Extract bounding box coordinates (xmin, ymin, width, height)
            xmin, ymin, width, height = d['bbox']
            xmax, ymax = xmin + width, ymin + height
            category_id = d['category_id']
            score = d['score']

            # Check if the center of the bounding box is inside the ROI polygon
            bbox_center = Point((xmin + xmax) / 2, (ymin + ymax) / 2)

            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))

            if category_id in allowed_class_ids:
                if roi_polygon_straight_and_right.contains(bbox_center):
                    vehicle_count_straight_and_right += 1
                    # yolo sahi detection 결과를 노란색 박스로 그리기
                    cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, f'{category_id}({score:.2f})', (int(xmin), int(ymin - 3)), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1) # {score:.2f}
                elif roi_polygon_left.contains(bbox_center):
                    vehicle_count_left += 1 
                    # yolo sahi detection 결과를 노란색 박스로 그리기
                    cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, f'{category_id}({score:.2f})', (int(xmin), int(ymin - 3)), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1) # {score:.2f}
                else:
                    # yolo sahi detection 결과를 노란색 박스로 그리기
                    cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=(100, 255, 255), thickness=1)
                    cv2.putText(frame, f'{category_id}({score:.2f})', (int(xmin), int(ymin - 3)), cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 255, 255), 1) # {score:.2f}
                    


        total_vehicle_count = vehicle_count_straight_and_right + vehicle_count_left
        cv2.putText(frame, f"{cam_direction} : {total_vehicle_count} vehicles", (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"sum_cnt_SR: {vehicle_count_straight_and_right}, sum_cnt_L: {vehicle_count_left}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.4, (128, 255, 128), 1)

        
        resized_image = cv2.resize(frame, (522, 696))  # (522, 696)
        return total_vehicle_count, vehicle_count_straight_and_right, vehicle_count_left, resized_image



    def get_carla_frame(self, cam_direction: CAM_DIRECTION):
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

    # def get_carla_status_by_image(self, cam_direction: CAM_DIRECTION, frame_interval: int):        
    #     return

    # def get_carla_status_by_numbers(self, image_array: np.ndarray, cam_direction: CAM_DIRECTION, frame_interval: int):
    #     stats = []
        
    #     frame = self.get_carla_frame(cam_direction)
        
    #     frame, roi_polygon_points = self.preprocess_frame(cam_direction, frame)
    #     roi_polygon = Polygon(roi_polygon_points)

    #     # Draw the ROI on the frame (optional, for visualization)
    #     cv2.polylines(frame, [roi_polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)
        
    #     # Perform YOLOv8 + SAHI detection on each frame
    #     result = get_sliced_prediction(
    #         image=frame, # img_path
    #         detection_model = model,
    #         slice_height=650, # 770  # 256
    #         slice_width=650,  # 770  # 256
    #         overlap_height_ratio=0.25, # 0.25
    #         overlap_width_ratio=0.25,  # 0.25
    #         postprocess_type="GREEDYNMM",  # Use NMS for handling overlapping boxes  # originally: GREEDYNMM
    #         postprocess_match_metric="IOS",  # Use IoU as the matching metric  # originally: IOS
    #         postprocess_match_threshold=0.3,  # Lower threshold to handle nested boxes  # originally: 0.5
    #         postprocess_class_agnostic=False
    #     )
    #     # result = get_prediction(frame, model)
    #     # result = model(frame)
    #     detections0 = result.to_coco_annotations()
        
    #     # Collect all detected objects
    #     detections1 = result.object_prediction_list
        
    #     # print(detections0)
    #     # print(detections1)
        
    #     # print(f"<< frame NO. {frame_count} >>")
    #     print(f"<< CAMERA {cam_direction} Object Detection >>")

    #     # Filter detections based on whether they are inside the ROI
    #     detection_results = []
    #     for d in detections0:
    #         # Extract bounding box coordinates (xmin, ymin, width, height)
    #         xmin, ymin, width, height = d['bbox']
    #         xmax, ymax = xmin + width, ymin + height

    #         # Check if the center of the bounding box is inside the ROI polygon
    #         bbox_center = Point((xmin + xmax) / 2, (ymin + ymax) / 2)
    #         if roi_polygon.contains(bbox_center) and d['category_id'] in allowed_class_ids:
    #             detection_results.append([xmin, ymin, xmax, ymax, d['score'], d['category_id']])
                
    #     stats.append(self.count_vehicles(detection_results))

    #     detection_results = np.array(detection_results)  # Convert list to NumPy array

    #     online_targets = []
        
    #     # Check the shape of detection_results
    #     if detection_results.size > 0:
    #         # Apply ByteTrack for tracking only within ROI
    #         online_targets = self.byte_trackers_state[cam_direction.value].update(  #         online_targets = byte_trackers[cam_direction].update(
    #             output_results=detection_results, 
    #             img_info=[frame.shape[0], frame.shape[1]], 
    #             img_size=[frame.shape[0], frame.shape[1]]
    #             )
    #         # else:
    #         #     print("Detection results do not have the expected shape.")
    #         #     online_targets = []  # Handle unexpected shape        

    #     print(online_targets)
        
    #     # Loop through tracked objects and draw bounding boxes with track IDs
    #     for track in online_targets:
    #         track_id = track.track_id
    #         score = track.score
    #         bbox = [int(i) for i in track.tlbr] # (top, left, width, height)  # Convert to int for drawing
    #         # bbox_center = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)       
    #         bbox_center_x, bbox_center_y  = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    #         lasted_frames = (track.end_frame-track.start_frame)*frame_interval+1

    #         print(f"[track ID {track_id}] location: {bbox} started: frame {(track.start_frame-1)*frame_interval}, duration: lasted {lasted_frames} frames")
    #         stats.extend([bbox_center_x, bbox_center_y, lasted_frames])
            
    #         if (len(stats)==1+90):
    #             break
            
            
    #         # Draw bounding box and track ID if inside ROI
    #         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    #         cv2.putText(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), 1) # {score:.2f}
            
    #     for i in range(91-len(stats)):
    #         stats.extend([0])
            
    #     print(f"len(stats): {len(stats)}")
                
    #     print()

    #     # # # # Optionally display the result
    #     # cv2.imshow('Frame', frame)
    #     # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     #     # out.release()
    #     #     cv2.destroyAllWindows()
        

    #     return stats  # returns list of length 91 (total 91 numbers, with '0' paddings)
    #                 # stats = [vehicle_count, bbox_center_x, bbox_center_y, lasted_frames, 
    #                 #                         bbox_center_x, bbox_center_y, lasted_frames, 
    #                 #                         ..., 
    #                 #                         bbox_center_x, bbox_center_y, lasted_frames, 
    #                 #                                                              0, ..., 0]
        

    # # Release everything if the job is done
    # out.release()
    # cv2.destroyAllWindows()



    def calculate_lasted_frames(self, frame: np.ndarray, cam_direction: CAM_DIRECTION, frame_interval: int, currentAction: int, yellowFlag: bool, prevAction: int):
        ## 1. return the total number of cars in 4 ROIs
        ## 2. return the total lasting times(frames) of cars in 4 ROIs
        
        # frame = get_carla_frame(cam_direction)
        
        # print("\t\t>> preprocess_frame...", end="")
        # frame, roi_polygon_points_straight_and_right, roi_polygon_points_left = self.preprocess_frame(cam_direction, image_array)
        roi_polygon_points_whole_lanes = np.array([(255, 26),(334, 26),(598, 516),(598, 799),(1, 800),(2, 370)])        
        roi_polygon_whole_lanes = Polygon(roi_polygon_points_whole_lanes)

        # Draw the ROI on the frame (optional, for visualization)
        cv2.polylines(frame, [roi_polygon_points_whole_lanes], isClosed=True, color=(255, 0, 0), thickness = 2)
        # cv2.putText(frame, f'ROI:SR', (258, 77 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 0, 0), 1) # {score:.2f}

        # cv2 uses (B,G,R), while numpy uses (R,G,B)
        R = (0,0,255)
        Y = (0,198,255)
        G = (0,181,82)

        traffic_sig_color_NS_pattern = [(G,G,R),(R,R,G),(R,R,R),(R,R,R),(R,R,R)]
        traffic_sig_color_EW_pattern = [(R,R,R),(R,R,R),(G,G,R),(R,R,G),(R,R,R)]
        

        if (cam_direction == CAM_DIRECTION.NORTH or cam_direction == CAM_DIRECTION.SOUTH):
            cv2.rectangle(frame, (1, 764), (141, 799), traffic_sig_color_NS_pattern[currentAction][0], -1)
            cv2.rectangle(frame, (142, 764), (419, 799), traffic_sig_color_NS_pattern[currentAction][1], -1)
            cv2.rectangle(frame, (420, 764), (598, 799), traffic_sig_color_NS_pattern[currentAction][2], -1)
        elif (cam_direction == CAM_DIRECTION.EAST or cam_direction == CAM_DIRECTION.WEST):
            cv2.rectangle(frame, (1, 764), (141, 799), traffic_sig_color_EW_pattern[currentAction][0], -1)
            cv2.rectangle(frame, (142, 764), (419, 799), traffic_sig_color_EW_pattern[currentAction][1], -1)
            cv2.rectangle(frame, (420, 764), (598, 799), traffic_sig_color_EW_pattern[currentAction][2], -1)


        if yellowFlag==True:
            if (prevAction == 0 and (cam_direction == CAM_DIRECTION.NORTH or cam_direction == CAM_DIRECTION.SOUTH)):
                cv2.rectangle(frame, (1, 764), (141, 799), Y, -1)
                cv2.rectangle(frame, (142, 764), (419, 799), Y, -1)
                # cv2.rectangle(frame, (420, 764), (598, 799), Y, -1)
            elif (prevAction == 1 and (cam_direction == CAM_DIRECTION.NORTH or cam_direction == CAM_DIRECTION.SOUTH)):
                # cv2.rectangle(frame, (1, 764), (141, 799), Y, -1)
                # cv2.rectangle(frame, (142, 764), (419, 799), Y, -1)
                cv2.rectangle(frame, (420, 764), (598, 799), Y, -1)
            elif (prevAction == 2 and (cam_direction == CAM_DIRECTION.EAST or cam_direction == CAM_DIRECTION.WEST)):
                cv2.rectangle(frame, (1, 764), (141, 799), Y, -1)
                cv2.rectangle(frame, (142, 764), (419, 799), Y, -1)
                # cv2.rectangle(frame, (420, 764), (598, 799), Y, -1)
            elif (prevAction == 3 and (cam_direction == CAM_DIRECTION.EAST or cam_direction == CAM_DIRECTION.WEST)):
                # cv2.rectangle(frame, (1, 764), (141, 799), Y, -1)
                # cv2.rectangle(frame, (142, 764), (419, 799), Y, -1)
                cv2.rectangle(frame, (420, 764), (598, 799), Y, -1)

        # (1, 764),(141, 799),
        # (142, 764),(419, 799),
        # (420, 764),(598, 799)




        # Perform YOLOv8 + SAHI detection on each frame
        #print("\t\t>> get_sliced_prediction...", end="")
        # result = get_sliced_prediction(
        #     image=frame, # img_path
        #     detection_model = model,
        #     slice_height=500, # 770 # 256
        #     slice_width=600,  # 770 # 256
        #     overlap_height_ratio=0.25, # 0.25
        #     overlap_width_ratio=0.25,  # 0.25
        #     postprocess_type="GREEDYNMM",  # Use NMS for handling overlapping boxes  # originally: GREEDYNMM
        #     postprocess_match_metric="IOS",  # Use IoU as the matching metric  # originally: IOS
        #     postprocess_match_threshold=0.2,  # Lower threshold to handle nested boxes  # originally: 0.5
        #     postprocess_class_agnostic=False,
        #     verbose=0
        # )
        result = get_prediction(frame, model)
        # result = model(frame)

        detections = result.to_coco_annotations()        
        
        # Filter detections based on whether they are inside the ROI
        detection_results = []
        # print("\t\t>> for d in detections0:...", end="")
        
        for object in detections:
            # Extract bounding box coordinates (xmin, ymin, width, height)
            xmin, ymin, width, height = object['bbox']
            xmax, ymax = xmin + width, ymin + height

            # Check if the center of the bounding box is inside the ROI polygon
            bbox_center = Point((xmin + xmax) / 2, (ymin + ymax) / 2)
            if roi_polygon_whole_lanes.contains(bbox_center) and (object['category_id'] in allowed_class_ids):
                detection_results.append([xmin, ymin, xmax, ymax, object['score'], object['category_id']])
                
                # Convert bounding box coordinates to integers
                pt1 = (int(xmin) + 10, int(ymin) + 10)
                pt2 = (int(xmax) - 10, int(ymax) - 10)

                # # yolo sahi detection 결과를 노란색 박스로 그리기
                # cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=(100, 255, 255), thickness=1)
                # cv2.putText(frame, 'det', (pt2[0], pt2[1] + 3), cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 255, 255), 1)
                
        cv2.putText(frame, f"{cam_direction.name}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)

        detection_results = np.array(detection_results)  # Convert list to NumPy array

        online_targets = []
        # Check the shape of detection_results
        if detection_results.size > 0:
            # Apply ByteTrack for tracking only within ROI
            online_targets = self.byte_trackers_reward[cam_direction.value].update(  #         online_targets = byte_trackers[cam_direction].update(
                output_results=detection_results, 
                img_info=[frame.shape[0], frame.shape[1]], 
                img_size=[frame.shape[0], frame.shape[1]]
            )

        sum_lasted_frames = 0
        
        # Loop through tracked objects and draw bounding boxes with track IDs
        for track in online_targets:
            track_id = track.track_id
            score = track.score
            bbox = [int(i) for i in track.tlbr] # (top, left, width, height)  # Convert to int for drawing
            # bbox_center = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)       
            bbox_center_x, bbox_center_y  = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

            ############################################################################
            lasted_frames = ( track.end_frame - track.start_frame ) * frame_interval + 1
            ############################################################################

            # Check if the center of the bounding box is inside the ROI polygon
            bbox_center = Point(bbox_center_x, bbox_center_y)

            sum_lasted_frames += lasted_frames

            # Draw bounding box and track ID
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 170), 2)
            text_size, _ = cv2.getTextSize('ID{track_id}({lasted_frames}fr)', cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            textlen = len('ID{track_id}({lasted_frames}fr)')/10
            cv2.rectangle(frame, (bbox[0],bbox[1]), (int(bbox[0]+textlen*40-15), int(bbox[1]-text_size[1]*1.2)), (0,0,0,0.5), -1)

            cv2.putText(frame, f'ID{track_id}({lasted_frames}fr)', (bbox[0], bbox[1] - 3), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1) # {score:.2f}


        cv2.putText(frame, f"sum_lasted_frs: {sum_lasted_frames}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)


        resized_image = cv2.resize(frame, (513, 684))

        return sum_lasted_frames, resized_image
        
        
        # print()

        # if cam_direction == CAM_DIRECTION.NORTH:
            # # Optionally display the result
        
        # cv2.imshow('Frame', frame)                        ################  <<=====   ##############
        
        # # image = Image.fromarray(frame)
        # # # image.save(f"{cam_direction}.png")
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     #out.release()
        #     cv2.destroyAllWindows()

        # total_average_waiting_time = sum_total_waiting_time / sum_waiting_vehicle_count if sum_waiting_vehicle_count > 0 else 0
        # reward = -1 * total_average_waiting_time   # 모든 대기차량의 평균 대기시간

        # return reward  # returns list of length 2
                    # reward = [vehicle_count, sum_lasted_frame]
        


    # # Release everything if the job is done
    # out.release()
    # cv2.destroyAllWindows()





 

