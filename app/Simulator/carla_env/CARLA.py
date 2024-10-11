import glob
import os
import sys
import time
# import random
import torch
from .enums import CAM_DIRECTION
from .object_detection import *


try:
    sys.path.append(glob.glob('C:/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
import numpy as np
from numpy import random

from collections import deque
import cv2

TICK_NUM = 100
GLOBAL_TICK_LIMIT = 300

MAX_VEHICLE_NUM = 200
RANDOM_SPAWN_NUM = 1

TRAFFIC_TICK_NUM = 70

IM_WIDTH = 600
IM_HEIGHT = 800

EPISODES = 5
EPISODE_TIME = 300
STEP_DELAY_TIME = 0.1
INITIAL_STEP = 10

class Env:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.synchronous_master = True        
        self.settings.fixed_delta_seconds = 0.5  # 0.2
        # self.settings.no_rendering_mode = True
        self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)
        
        self.traffic_manager = self.client.get_trafficmanager() # 8000
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_random_device_seed(0)
        random.seed(0)
        self.traffic_manager.set_respawn_dormant_vehicles(False)
        
        self.SpawnActor = carla.command.SpawnActor
        self.SetAutopilot = carla.command.SetAutopilot
        self.FutureActor = carla.command.FutureActor
        
        self.target_actor_lists = [
            [1, 2, 4, 5, 6, 7],
            [0, 3],
            [10, 11, 12, 13, 14, 15],
            [8, 9]
        ]
        
        self.camera_positions = [
            [-18, 5, 7, -50, 180, 0],  # Camera 1
            [5, 18, 7, -50, 90, 0],  # Camera 2
            [18, -5, 7, -50, 0, 0],  # Camera 3
            [-5, -18, 7, -50, -90, 0]   # Camera 4
        ]
        self.frame = 0
        
        self.blueprints = []
        self.blueprints = self.get_actor_blueprints(self.world, 'vehicle.*', 'All')
        self.blueprints = [x for x in self.blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        self.blueprints = [x for x in self.blueprints if x.get_attribute('base_type') == 'car']
        self.blueprints = sorted(self.blueprints, key=lambda bp: bp.id)
        
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.number_of_vehicles = 40
        self.max_vehicles = 100
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.number_of_spawn_points = len(self.spawn_points)
        if self.number_of_vehicles < self.number_of_spawn_points:
            random.shuffle(self.spawn_points)
        else:
            print("number_of_vehicles > number_of_spawn_points")
            
        self.im_width = IM_WIDTH
        self.im_height = IM_HEIGHT

        self.cameras = []
        self.camera_images = [None, None, None, None]
        
        for i, pos in enumerate(self.camera_positions):
            cam_bp = self.blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute("image_size_x", f"{self.im_width}")
            cam_bp.set_attribute("image_size_y", f"{self.im_height}")
            cam_bp.set_attribute("fov", "90")
            
            transform = carla.Transform(carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                                        carla.Rotation(pitch=pos[3], yaw=pos[4], roll=pos[5]))
            camera = self.world.spawn_actor(cam_bp, transform)
            # print(f"\t\t>> camera_{i} : spawned")
            self.cameras.append(camera)
            camera.listen(lambda data, idx=i: self.process_image(data, idx))
        
    def main(self):
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.synchronous_master = True        
        self.settings.fixed_delta_seconds = 0.5
        # self.settings.no_rendering_mode = True
        self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)
        
        self.vehicles_list = []
        self.list_traffic_light_actor = []
        
        self.elapsed_global_tick = 0
        self.camera_images = [None, None, None, None]
        
        # --------------
        # Spawn vehicles
        # --------------
        # 차량을 생성하는 부분 수정
        self.batch = []
        for n, self.transform in enumerate(self.spawn_points):
            if n >= self.number_of_vehicles:
                break
            self.blueprint = random.choice(self.blueprints)
            if self.blueprint.has_attribute('color'):
                self.color = random.choice(self.blueprint.get_attribute('color').recommended_values)
                self.blueprint.set_attribute('color', self.color)
            if self.blueprint.has_attribute('driver_id'):
                self.driver_id = random.choice(self.blueprint.get_attribute('driver_id').recommended_values)
                self.blueprint.set_attribute('driver_id', self.driver_id)
            self.blueprint.set_attribute('role_name', 'autopilot')
            
            # 차량을 생성하고 batch에 추가
            self.batch.append(self.SpawnActor(self.blueprint, self.transform).then(self.SetAutopilot(self.FutureActor, True)))

        # batch를 실행하여 차량을 생성
        for self.response in self.client.apply_batch_sync(self.batch, self.synchronous_master):
            if self.response.error:
                logging.error(self.response.error)
            else:
                self.vehicles_list.append(self.response.actor_id)
                
        print("\t\t>> getting Traffic Light Actors...")

        self.getTrafficLightActor()
        
        print("\t\t>> succesful..")
        
    def reset(self):
        print(">> clean up episode..")
        
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = False
        self.settings.no_rendering_mode = True
        self.settings.fixed_delta_seconds = 0.5 # 0.3
        self.world.apply_settings(self.settings)
            
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.vehicles_list = []
        self.list_traffic_light_actor = []
        self.camera_images = [None, None, None, None]
        self.batch = []
        
        time.sleep(0.5)
        
    def process_image(self, image, camera_index):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        self.camera_images[camera_index] = i3
     
    def get_camera_images(self):
        print(f"\t\t\t>> get_camera_images.. <<")
        return self.camera_images
        
    def get_images(self):
        img_array = []
        images = self.get_camera_images()
        idx = 0
        for img in images:
            if img is not None:
                img_array.append(img)
                # cv2.imwrite(f"_out/camera_{idx+1}_frame_{self.frame+1}.png", img)
                idx += 1
                self.frame += 1
                time.sleep(0.03)
        img_array = np.uint8(images)
        # time.sleep(0.03)

        return img_array
        
    def get_actor_blueprints(self, world, filter, generation):
        self.bps = world.get_blueprint_library().filter(filter)

        if generation.lower() == "all":
            return self.bps

        if len(self.bps) == 1:
            return self.bps

        try:
            self.int_generation = int(generation)
            if self.int_generation in [1, 2, 3]:
                self.bps = [x for x in self.bps if int(x.get_attribute('generation')) == self.int_generation]
                return self.bps
            else:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []
        except:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []

    ###############
    #   Monitor   #
    ###############
    
    def getTrafficLightActor(self):
        self.list_actor = self.world.get_actors()
        # print(f">> get {self.list_actor}")
        for actor_ in self.list_actor:
            if isinstance(actor_, carla.TrafficLight) and actor_.type_id == "traffic.traffic_light":
                self.list_traffic_light_actor.append(actor_)
                print(f"\t\t>> '{actor_.type_id}' : appended into list_traffic_ligth_actor")
                # time.sleep(0.01)

    def setTrafficLightGreen(self, target_actor):
        target_actor.set_state(carla.TrafficLightState.Green)
        print(f"\t\t>> {target_actor} : now green")

    def setTrafficLightRed(self, target_actor):
        target_actor.set_state(carla.TrafficLightState.Red)
        print(f"\t\t>> {target_actor} : now red")

    def setTrafficLightGroupGreen(self, target_actors):
        for i in target_actors:
            self.setTrafficLightGreen(self.list_traffic_light_actor[i])
            print(f"\t\t>> {i} : now green")
        
    def setTrafficLightGroupRed(self, target_actors):
        for i in target_actors:
            self.setTrafficLightRed(self.list_traffic_light_actor[i])
            print(f"\t\t>> {i} : now red")

    def allRed(self):
        for actor_ in self.list_actor:
            if isinstance(actor_, carla.TrafficLight):
                actor_.set_state(carla.TrafficLightState.Red)
                actor_.set_red_time(1000.0)
        print(f"\t\t>> now all red..")
        
    def setTrafficLightPattern0(self):
        self.setTrafficLightGroupGreen(self.target_actor_lists[0])

    def setTrafficLightPattern1(self):
        self.setTrafficLightGroupGreen(self.target_actor_lists[1])

    def setTrafficLightPattern2(self):
        self.setTrafficLightGroupGreen(self.target_actor_lists[2])

    def setTrafficLightPattern3(self):
        self.setTrafficLightGroupGreen(self.target_actor_lists[3])

    ##############
    #    Tick    #
    ##############

    def Tick(self, tick_num = TICK_NUM):
        elapsed_tick = 0
        while elapsed_tick < tick_num:
            print(f"\t\t>> tick : {elapsed_tick}\t\t>> global_tick : {self.elapsed_global_tick}")
            self.world.tick()
            self.elapsed_global_tick += 1
            elapsed_tick += 1

    def spawnRandom(self):
        print("spawn??")
        self.batch = []
        # 현재 스폰된 차량의 수가 최대 차량 수를 넘지 않도록 체크
        if len(self.vehicles_list) >= self.max_vehicles:
            print(f"Maximum number of vehicles ({self.max_vehicles}) already spawned.")
            return
        
        # 무작위로 선택된 spawn_points에 새로운 차량을 스폰
        for i, targetSpawnPoint in enumerate(np.random.choice(self.spawn_points, RANDOM_SPAWN_NUM, False)):
            self.blueprint = random.choice(self.blueprints)
            
            if self.blueprint.has_attribute('color'):
                self.color = random.choice(self.blueprint.get_attribute('color').recommended_values)
                self.blueprint.set_attribute('color', self.color)
            
            self.blueprint.set_attribute('role_name', 'autopilot')
            
            # 차량을 스폰하고 리스트에 추가
            self.batch.append(self.SpawnActor(self.blueprint, targetSpawnPoint)
                .then(self.SetAutopilot(self.FutureActor, True, self.traffic_manager.get_port())))

        # 스폰된 차량 목록을 업데이트
        for self.response in self.client.apply_batch_sync(self.batch, self.synchronous_master):
            if self.response.error:
                logging.error(self.response.error)
            else:
                self.vehicles_list.append(self.response.actor_id)
        
        print(f"Spawned {RANDOM_SPAWN_NUM} vehicle(s), Total: {len(self.vehicles_list)} vehicles.")

class CARLA:
    def __init__(self):

        self.a = 1.005
        self.lasted_frame_north = 0
        self.lasted_frame_east = 0
        self.lasted_frame_south = 0
        self.lasted_frame_west = 0

        self.lastAction = -1
        self.env = Env()
        self.ObjectTracking = ObjectTracking()

    def start(self):
        # 시뮬레이터 환경설정 및 가동
        # self.env.tick()
        self.env.main()

        self.lastAction = -1

        self.lasted_frame_north = 0
        self.lasted_frame_east = 0
        self.lasted_frame_south = 0
        self.lasted_frame_west = 0

        self.ObjectTracking.start()

        print(f"\t\t\t>> start.. <<")
        self.env.world.tick(50)
        self.env.allRed()

        state = self.get_state()
        return state

    def step(self, action):
        if self.lastAction == -1:
            pass

        elif self.lastAction != action:
            self.env.allRed()
            for i in range(14):
                self.env.Tick(5)
                self.updateLastedFrame()

        if action == 0:
            print(f"\t\t\t\t\t\t>>>> {self.lastAction} --> Traffic Control 00 <<<<")
            self.env.setTrafficLightPattern0()

        elif action == 1:
            print(f"\t\t\t\t\t\t>>>> {self.lastAction} --> Traffic Control 01 <<<<")
            self.env.setTrafficLightPattern1()

        elif action == 2:
            print(f"\t\t\t\t\t\t>>>> {self.lastAction} --> Traffic Control 02 <<<<")
            self.env.setTrafficLightPattern2()

        elif action == 3:
            print(f"\t\t\t\t\t\t>>>> {self.lastAction} --> Traffic Control 03 <<<<")
            self.env.setTrafficLightPattern3()

        self.env.spawnRandom()

        self.lastAction = action

        next_state = self.get_state()
        reward = -((self.a ** self.lasted_frame_north) + (self.a ** self.lasted_frame_east) + (self.a ** self.lasted_frame_south) + (self.a ** self.lasted_frame_west))
        done = self.is_done()
        
        return next_state, reward, done
    
    def get_state(self):
        print(f"\t\t\t>> get_state.. <<")

        image_tensors = []

        for i in range(4):

            # 24fps 기준 12프레임 진행
            self.env.Tick(5)
            self.updateLastedFrame()

            # 십자 교차로 기준 4가지 뷰에 대한 화면 이미지 로드
            image_array = self.env.get_images()
            # print(image_array)

            # 4개의 이미지를 4채널 tensor로 변환
            for image in image_array:
                img_tensor = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1)
                image_tensors.append(img_tensor)

        # 16채널 tensor (channels, height, width)
        state = torch.cat(image_tensors, dim=0)
        # print(state.size())
        # print(state)
        return state
    
    def updateLastedFrame(self):

        image_array = []
        image_array = self.env.get_images()         
         # Implement the logic to calculate the reward
        # print(type(image_array[0]))

        lasted_frame_list = [self.lasted_frame_north, self.lasted_frame_east, self.lasted_frame_south, self.lasted_frame_west]

        i = 0
        for cam_direction in CAM_DIRECTION:
            # print(f"caculate_reward_from_carla {i}...", end="")
            # time.sleep(0.1)
            vehicle_count, sum_lasted_frame = self.ObjectTracking.calculate_reward_from_carla(image_array[i], cam_direction=cam_direction, frame_interval=5)
            # time.sleep(0.1)
            # print("ok")
            lasted_frame_list[i] = sum_lasted_frame
            i+=1

        print(f"lasted_frame_list: {lasted_frame_list}")

    
    def is_done(self):
        if self.env.elapsed_global_tick > GLOBAL_TICK_LIMIT:
            return True
        else:
            return False
    
    def reset(self):
        self.env.reset()


if __name__ == "__main__":
    # try:
    #     env = CARLA()
    #     env.start()
        
    #     env.step(0)
    #     env.step(0)
    #     env.step(0)
    #     env.step(1)
    #     env.step(1)
    #     env.step(1)
    #     env.step(2)
    #     env.step(2)
    #     env.step(2)
    #     env.step(3)
    #     env.step(3)
    #     env.step(3)
 
    #     print("done!!")

    # except:
    #     KeyboardInterrupt
    # finally:
    #     env.reset()
        

    env = CARLA()
    
    env.start()
    
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(4)
    env.step(4)
    env.step(4)
    env.step(4)
    env.step(4)
    env.step(4)
    
    env.reset()
    print("episode 0 done")
    
    env.start()
    
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(4)
    env.step(4)
    env.step(4)
    env.step(4)
    env.step(4)
    env.step(4)
    
    env.reset()    
    print("episode 1 done")
    
    env.start()
    
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(4)
    env.step(4)
    env.step(4)
    env.step(4)
    env.step(4)
    env.step(4)
    
    env.reset()
    print("episode 2 done")
    
    try:
        while True:
            env.env.world.tick()
    except:
        KeyboardInterrupt
    finally:
        print("done")