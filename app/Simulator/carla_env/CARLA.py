import glob
import os
import sys
import time
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

logger = logging.getLogger('ftpuploader')
hdlr = logging.FileHandler('ftplog.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
FTPADDR = "some ftp address"

TICK_NUM = 100
GLOBAL_TICK_LIMIT = 200

TRAFFIC_TICK_NUM = 70

IM_WIDTH = 600
IM_HEIGHT = 800

EPISODES = 5
EPISODE_TIME = 300
STEP_DELAY_TIME = 0.1
INITIAL_STEP = 10

class Env:
    def __init__(self):

        self.start_time = time.time()
        
        self.argparser = argparse.ArgumentParser(
            description=__doc__)
        self.argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        self.argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        self.argparser.add_argument(
            '-n', '--number-of-vehicles',
            metavar='N',
            default=80,        # number of vehicles
            type=int,
            help='Number of vehicles (default: 30)')
        self.argparser.add_argument(
            '-w', '--number-of-walkers',
            metavar='W',
            default=10,
            type=int,
            help='Number of walkers (default: 10)')
        self.argparser.add_argument(
            '--safe',
            action='store_true',
            help='Avoid spawning vehicles prone to accidents')
        self.argparser.add_argument(
            '--filterv',
            metavar='PATTERN',
            default='vehicle.*',
            help='Filter vehicle model (default: "vehicle.*")')
        self.argparser.add_argument(
            '--generationv',
            metavar='G',
            default='All',
            help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
        self.argparser.add_argument(
            '--filterw',
            metavar='PATTERN',
            default='walker.pedestrian.*',
            help='Filter pedestrian type (default: "walker.pedestrian.*")')
        self.argparser.add_argument(
            '--generationw',
            metavar='G',
            default='2',
            help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
        self.argparser.add_argument(
            '--tm-port',
            metavar='P',
            default=8000,
            type=int,
            help='Port to communicate with TM (default: 8000)')
        self.argparser.add_argument(
            '--asynch',
            action='store_true',
            help='Activate asynchronous mode execution')
        self.argparser.add_argument(
            '--hybrid',
            action='store_true',
            help='Activate hybrid mode for Traffic Manager')
        self.argparser.add_argument(
            '-s', '--seed',
            metavar='S',
            type=int,
            help='Set random device seed and deterministic mode for Traffic Manager')
        self.argparser.add_argument(
            '--seedw',
            metavar='S',
            default=0,
            type=int,
            help='Set the seed for pedestrians module')
        self.argparser.add_argument(
            '--car-lights-on',
            action='store_true',
            default=False,
            help='Enable automatic car light management')
        self.argparser.add_argument(
            '--hero',
            action='store_true',
            default=False,
            help='Set one of the vehicles as hero')
        self.argparser.add_argument(
            '--respawn',
            action='store_true',
            default=False,
            help='Automatically respawn dormant vehicles (only in large maps)')
        self.argparser.add_argument(
            '--no-rendering',
            action='store_true',
            default=False,
            help='Activate no rendering mode')

        self.args = self.argparser.parse_args()

        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)
        self.synchronous_master = False
        random.seed(self.args.seed if self.args.seed is not None else int(time.time()))

    def main(self):
        self.world = self.client.get_world()

        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.elapsed_global_tick = 0

        self.traffic_manager = self.client.get_trafficmanager(self.args.tm_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if self.args.respawn:
            self.traffic_manager.set_respawn_dormant_vehicles(True)
        if self.args.hybrid:
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(70.0)
        if self.args.seed is not None:
            self.traffic_manager.set_random_device_seed(self.args.seed)

        self.settings = self.world.get_settings()
        if not self.args.asynch:
            self.traffic_manager.set_synchronous_mode(True)
            if not self.settings.synchronous_mode:
                self.synchronous_master = True
                self.settings.synchronous_mode = True
                self.settings.fixed_delta_seconds = 0.3     # set delta second
            else:
                self.synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        if self.args.no_rendering:
            self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)

        self.blueprints = self.get_actor_blueprints(self.world, self.args.filterv, self.args.generationv)
        if not self.blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")
        self.blueprintsWalkers = self.get_actor_blueprints(self.world, self.args.filterw, self.args.generationw)
        if not self.blueprintsWalkers:
            raise ValueError("Couldn't find any walkers with the specified filters")

        if self.args.safe:
            self.blueprints = [x for x in self.blueprints if x.get_attribute('base_type') == 'car']

        self.blueprints = sorted(self.blueprints, key=lambda bp: bp.id)

        self.spawn_points = self.world.get_map().get_spawn_points()
        self.number_of_spawn_points = len(self.spawn_points)

        if self.args.number_of_vehicles < self.number_of_spawn_points:
            random.shuffle(self.spawn_points)
        elif self.args.number_of_vehicles > self.number_of_spawn_points:
            self.msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(self.msg, self.args.number_of_vehicles, self.number_of_spawn_points)
            self.args.number_of_vehicles = self.number_of_spawn_points

        # @todo cannot import these directly.
        self.SpawnActor = carla.command.SpawnActor
        self.SetAutopilot = carla.command.SetAutopilot
        self.FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        self.batch = []
        self.hero = self.args.hero
        for n, self.transform in enumerate(self.spawn_points):
            if n >= self.args.number_of_vehicles:
                break
            self.blueprint = random.choice(self.blueprints)
            if self.blueprint.has_attribute('color'):
                self.color = random.choice(self.blueprint.get_attribute('color').recommended_values)
                self.blueprint.set_attribute('color', self.color)
            if self.blueprint.has_attribute('driver_id'):
                self.driver_id = random.choice(self.blueprint.get_attribute('driver_id').recommended_values)
                self.blueprint.set_attribute('driver_id', self.driver_id)
            if self.hero:
                self.blueprint.set_attribute('role_name', 'hero')
                self.hero = False
            else:
                self.blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            self.batch.append(self.SpawnActor(self.blueprint, self.transform)
                .then(self.SetAutopilot(self.FutureActor, True, self.traffic_manager.get_port())))

        for self.response in self.client.apply_batch_sync(self.batch, self.synchronous_master):
            if self.response.error:
                logging.error(self.response.error)
            else:
                self.vehicles_list.append(self.response.actor_id)

        # Set automatic vehicle lights update if specified
        if self.args.car_lights_on:
            self.all_vehicle_actors = self.world.get_actors(self.vehicles_list)
            for self.actor in self.all_vehicle_actors:
                self.traffic_manager.update_vehicle_lights(self.actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        self.percentagePedestriansRunning = 0.0      # how many pedestrians will run
        self.percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        if self.args.seedw:
            self.world.set_pedestrians_seed(self.args.seedw)
            random.seed(self.args.seedw)
        # 1. take all the random locations to spawn
        self.spawn_points = []
        for i in range(self.args.number_of_walkers):
            self.spawn_point = carla.Transform()
            self.loc = self.world.get_random_location_from_navigation()
            if (self.loc != None):
                self.spawn_point.location = self.loc
                self.spawn_points.append(self.spawn_point)
        # 2. we spawn the walker object
        self.batch = []
        self.walker_speed = []
        for self.spawn_point in self.spawn_points:
            self.walker_bp = random.choice(self.blueprintsWalkers)
            # set as not invincible
            self.probability = random.randint(0,100 + 1);
            if self.walker_bp.has_attribute('is_invincible'):
                self.walker_bp.set_attribute('is_invincible', 'false')
            if self.walker_bp.has_attribute('can_use_wheelchair') and self.probability < 11:
                self.walker_bp.set_attribute('use_wheelchair', 'true')
            # set the max speed
            if self.walker_bp.has_attribute('speed'):
                if (random.random() > self.percentagePedestriansRunning):
                    # walking
                    self.walker_speed.append(self.walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    self.walker_speed.append(self.walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                self.walker_speed.append(0.0)
            self.batch.append(SpawnActor(self.walker_bp, self.spawn_point))
        self.results = self.client.apply_batch_sync(self.batch, True)
        self.walker_speed2 = []
        for i in range(len(self.results)):
            if self.results[i].error:
                logging.error(self.results[i].error)
            else:
                self.walkers_list.append({"id": self.results[i].actor_id})
                self.walker_speed2.append(self.walker_speed[i])
        self.walker_speed = self.walker_speed2
        # 3. we spawn the walker controller
        self.batch = []
        self.walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            self.batch.append(self.SpawnActor(self.walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        self.results = self.client.apply_batch_sync(self.batch, True)
        for i in range(len(self.results)):
            if self.results[i].error:
                logging.error(self.results[i].error)
            else:
                self.walkers_list[i]["con"] = self.results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if self.args.asynch or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        self.world.set_pedestrians_cross_factor(self.percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
                # start walker
            self.all_actors[i].start()
                # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
                # max speed
            self.all_actors[i].set_max_speed(float(self.walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles_list), len(self.walkers_list)))

            # Example of how to use Traffic Manager parameters
        self.traffic_manager.global_percentage_speed_difference(30.0)

        self.list_traffic_light_actor = []

        self.target_actor_lists = [
            [1, 2, 4, 5, 6, 7],
            [0, 3],
            [10, 11, 12, 13, 14, 15],
            [8, 9]
        ]

        print("\t\t>> getting Traffic Light Actors...")

        self.getTrafficLightActor()
        
        print("\t\t>> succesful..")

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
                time.sleep(0.01)
        
    def setTrafficLightGreenTime(self):
        list_green_time = []
        
        # tmp for now.. TODO : make set list green time function
        for i in range(len(self.list_traffic_light_actor) + 1):
            list_green_time.append(i)
        
        idx = 0
        
        for traffic_light_actor in self.list_traffic_light_actor:
            
            traffic_light_actor.set_green_time(list_green_time[idx])
            idx = idx + 1
            print(f"\t\t>> '{traffic_light_actor}' : set green time '{traffic_light_actor.get_green_time()}' --> '{list_green_time[idx]}'")
            
    def setTrafficLightRedTime(self):
        list_red_time = []
        
        # tmp for now.. TODO : make set list red time function
        for i in range(len(self.list_traffic_light_actor) + 1):
            list_red_time.append(3)
        
        idx = 0
        for traffic_light_actor in self.list_traffic_light_actor:
            
            traffic_light_actor.set_red_time(list_red_time[idx])
            idx = idx + 1
            print(f"\t\t>> '{traffic_light_actor}' : set red time '{traffic_light_actor.get_red_time()}' --> '{list_red_time[idx]}'")
    
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

    def allGreen(self):
        for traffic_light_actor in self.list_traffic_light_actor:
            if isinstance(traffic_light_actor, carla.TrafficLight):
                traffic_light_actor.set_state(carla.TrafficLightState.Green)
                traffic_light_actor.set_green_time(1000.0)
        print(f"\t\t>> now all green..")

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

    def cleanUp(self):
        print(">> clean up episode..")
        if not self.args.asynch and self.synchronous_master:
            self.settings = self.world.get_settings()
            self.settings.synchronous_mode = False
            self.settings.no_rendering_mode = False
            self.settings.fixed_delta_seconds = None
            self.world.apply_settings(self.settings)

        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\ndestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        time.sleep(0.5)



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
            [-18, 5, 7, -50, 180, 0],  # Camera 1
            [5, 18, 7, -50, 90, 0],  # Camera 2
            [18, -5, 7, -50, 0, 0],  # Camera 3
            [-5, -18, 7, -50, -90, 0]   # Camera 4
        ]
        # print("done.")

        # self.camera_positions = [
        #     [-18, 8, 8, -50, 200, 0],  # Camera 1
        #     [8, 18, 8, -50, 110, 0],  # Camera 2
        #     [18, -8, 8, -50, 20, 0],  # Camera 3
        #     [-8, -18, 8, -50, -70, 0]   # Camera 4
        # ]
        # # print("done.")

    def setup_cameras(self):
        print(">> setting up cameras..")
        for i, pos in enumerate(self.camera_positions):
            cam_bp = self.blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute("image_size_x", f"{self.im_width}")
            cam_bp.set_attribute("image_size_y", f"{self.im_height}")
            cam_bp.set_attribute("fov", "90")
            
            transform = carla.Transform(carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                                        carla.Rotation(pitch=pos[3], yaw=pos[4], roll=pos[5]))
            camera = self.world.spawn_actor(cam_bp, transform)
            # print(f"\t\t>> camera_{i} : spawned")
            self.actor_list.append(camera)
            self.cameras.append(camera)
            camera.listen(lambda data, idx=i: self.process_image(data, idx))
        print("done.")

    def process_image(self, image, camera_index):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]  # RGB 값 추출
        # i3 = cv2.cvtColor(i3, cv2.COLOR_BGR2GRAY)  # 필요하다면 흑백 이미지로 변환
        # i3 = i3 / 255.0  # 정규화
        # print(i3)
        self.camera_images[camera_index] = i3
        # print(type(self.camera_images[camera_index])) # <numpy.ndarray>
        # print(self.camera_images[camera_index].shape) # (800, 600, 3)

    def reset(self):
        self.camera_images = [None, None, None, None]
        # Set up initial state if necessary

    def destroy_actors(self):
        """Destroy all actors."""
        print(">> destroying cameras..", end="\t")
        for actor in self.actor_list:
            actor.destroy()
        print("done.")        

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
                # print(f"\t\t>> get_images {idx} appended")
                # cv2.imwrite(f"_out/camera_{idx+1}_frame_{self.frame+1}.png", img)
                time.sleep(0.1)
                idx += 1
                self.frame += 1
        img_array = np.uint8(images)
        # time.sleep(0.05)
        # print(img_array.shape)
        # print(img_array)

        return img_array

class CARLA:
    def __init__(self):

        self.a = 1.001
        self.t_north = 0
        self.t_east = 0
        self.t_south = 0
        self.t_west = 0

        self.lastAction = -1


        self.env = Env()

        self.monitor = IntersectionMonitor()


    def start(self):
        # 시뮬레이터 환경설정 및 가동
        # self.env.tick()
        self.env.main()

        self.env.world.tick()

        self.monitor.setup_cameras()

        self.lastAction = -1


        self.t_north = 0
        self.t_east = 0
        self.t_south = 0
        self.t_west = 0

        print(f"\t\t\t>> start.. <<")
        self.env.allRed()

        state = self.get_state()
        return state
    
    ####################
    #   0 : Pattern0   #
    #   1 : Pattern1   #
    #   2 : Pattern2   #
    #   3 : Pattern3   #
    #   4 : allRed     #
    ####################

    def step(self, action):
        # 입력받은 action 값에 따른 신호제어

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

        self.lastAction = action

        next_state = self.get_state()
        reward = -((self.a ** self.t_north) + (self.a ** self.t_east) + (self.a ** self.t_south) + (self.a ** self.t_west))
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
            image_array = self.monitor.get_images()

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
        image_array = self.monitor.get_images()         
         # Implement the logic to calculate the reward
        # print(type(image_array[0]))

        t_list = [self.t_north, self.t_east, self.t_south, self.t_west]

        i = 0
        for cam_direction in CAM_DIRECTION:
            # print(f"caculate_reward_from_carla {i}...", end="")
            # time.sleep(0.1)
            vehicle_count, sum_lasted_frame = caculate_reward_from_carla(image_array[i], cam_direction=cam_direction, frame_interval=15)
            # time.sleep(0.1)
            # print("ok")
            t_list[i] = sum_lasted_frame
            i+=1

    
    def is_done(self):
        if self.env.elapsed_global_tick > GLOBAL_TICK_LIMIT:
            return True
        else:
            return False
    
    def reset(self):
        # self.monitor.destroy_actors()
        self.monitor.reset()
        self.env.cleanUp()


if __name__ == "__main__":
    try:
        env = CARLA()
        env.start()
        
        env.step(0)
        env.step(0)
        env.step(0)
        env.step(1)
        env.step(1)
        env.step(1)
        env.step(2)
        env.step(2)
        env.step(2)
        env.step(3)
        env.step(3)
        env.step(3)
 
        print("done!!")

    except Exception as e:
        logger.error('Failed to do something: ' + str(e))
    finally:
        env.reset()
        

    # env = CARLA()
    # env.start()
    
    # env.step(1)
    # env.step(1)
    # env.step(1)
    # env.step(2)
    # env.step(2)
    # env.step(2)