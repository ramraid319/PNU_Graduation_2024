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

import argparse
import logging
from numpy import random

traffic_light_01_name = "BP_TrafficLightNew_T10_master_largeBIG_rsc"
traffic_light_02_name = "BP_TrafficLightNew_T10_master_largeBIG_rsc2"
traffic_light_03_name = "BP_TrafficLightNew_T10_master_largeBIG_rsc3"
traffic_light_04_name = "BP_TrafficLightNew_T10_master_largeBIG_rsc4"


class carlaEnv:

    def __init__(self):
        print("\t\t>> initiate carlaEnv..", end=' ')
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.list_actor = self.world.get_actors()
        
        # wtf no getter using bp_name
        # self.list_traffic_light_actor_name = [traffic_light_01_name, traffic_light_02_name, traffic_light_03_name, traffic_light_04_name]
        
        self.list_traffic_light_actor = []
        
        print("succesful..")
    
    def getTrafficLightActor(self):
        for actor_ in self.list_actor:
            if isinstance(actor_, carla.TrafficLight) and actor_.type_id == "traffic.traffic_light":
                self.list_traffic_light_actor.append(actor_)
                print(f"\t\t>> '{actor_.type_id}' : appended into list_traffic_ligth_actor")
        
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
        print(f"\t\t>> now all green..")

    def set_traffic_light_state_by_name(self, name, state):
        # 모든 액터 가져오기
        actors = self.world.get_actors()
        
        # 이름으로 신호등 찾기
        for actor in actors:
            if isinstance(actor, carla.TrafficLight) and actor.get_display_name() == name:
                actor.set_state(state)
                actor.set_manual_control(True)  # 수동 제어 모드로 설정
                print(f"신호등 '{name}' 상태를 {state}로 설정했습니다.")
                return
        
        print(f"이름 '{name}'의 신호등을 찾을 수 없습니다.")

def main():
    # 예시: 추가적으로 필요한 동작을 main에 작성할 수 있습니다.
    print("running control_traffic_light..")
    
    Env = carlaEnv()
    
    Env.getTrafficLightActor()
    
    # Env.setTrafficLightGreenTime()
    # Env.setTrafficLightRedTime()
    
    # Env.allGreen()
    Env.allRed()

if __name__ == '__main__':
    main()
