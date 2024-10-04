import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import numpy as np
from numpy import random

from queue import Queue
from queue import Empty

import carla
import time
from collections import deque
import cv2

IM_WIDTH = 640
IM_HEIGHT = 480

class IntersectionMonitor:
    
    def __init__(self):
        print(">> init Monitor..", end="\t")
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Camera settings
        self.im_width = IM_WIDTH
        self.im_height = IM_HEIGHT
        self.cameras = []
        self.actor_list = []
        
        self.frame = 0

        # Store the images from each camera
        self.camera_images = [None, None, None, None]
        
        self.camera_positions = [
            [-10, 1, 20, -40, 180, 0],  # Camera 1
            [1, 10, 20, -40, 90, 0],  # Camera 2
            [10, 1, 20, -40, 0, 0],  # Camera 3
            [1, -10, 20, -40, -90, 0]   # Camera 4
        ]
        print("done.")

    def setup_cameras(self):
        """Set up four cameras at the given positions."""
        print(">> setting up cameras..", end="\t")
        for i, pos in enumerate(self.camera_positions):
            cam_bp = self.blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute("image_size_x", f"{self.im_width}")
            cam_bp.set_attribute("image_size_y", f"{self.im_height}")
            cam_bp.set_attribute("fov", "110")
            
            transform = carla.Transform(carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                                        carla.Rotation(pitch=pos[3], yaw=pos[4], roll=pos[5]))
            camera = self.world.spawn_actor(cam_bp, transform)
            self.actor_list.append(camera)
            self.cameras.append(camera)
            camera.listen(lambda data, idx=i: self.process_image(data, idx))
        print("done.")

    def process_image(self, image, camera_index):
        """Process the image from the camera and store it."""
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]  # RGB 값 추출
        # i3 = cv2.cvtColor(i3, cv2.COLOR_BGR2GRAY)  # 필요하다면 흑백 이미지로 변환
        # i3 = i3 / 255.0  # 정규화
        # print(i3)
        self.camera_images[camera_index] = i3

    def reset(self):
        """Reset the camera images and set up the environment."""
        self.camera_images = [None, None, None, None]
        # Set up initial state if necessary

    def destroy_actors(self):
        """Destroy all actors."""
        print(">> destroying cameras..", end="\t")
        for actor in self.actor_list:
            actor.destroy()
        print("done.")        

    def get_camera_images(self):
        """Return the latest images from all cameras."""
        return self.camera_images
    
    def save_images(self):
        images = self.get_camera_images()
        for idx, img in enumerate(images):
            if img is not None:
                cv2.imwrite(f"_out/camera_{idx+1}_frame_{self.frame+1}.png", img)
                self.frame += 1
        time.sleep(1)