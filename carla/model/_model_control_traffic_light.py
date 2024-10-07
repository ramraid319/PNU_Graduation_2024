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

class TrafficController:

    def __init__(self):
        print("\t\t>> initiate TrafficController..", end=' ')
        
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
        
    def setTrafficLightGroupGreen(self, target_actors):
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
        

def main():
    print("running control_traffic_light..")
    
    Controller = TrafficController()
    
    Controller.getTrafficLightActor()
    
    Controller.allRed()
    
    # Controller.setTrafficLightGreenTime()
    # Controller.setTrafficLightRedTime()
    
    # Controller.allGreen()
    # Controller.setTrafficLightGreen(Controller.list_traffic_light_actor[15])    
    # Controller.setTrafficLightGreen(Controller.list_traffic_light_actor[5])
    

    ############PATTERN GROUP############
    # GROUP 0 : 1, 2, 4, 5, 6, 7        #
    # GROUP 1 : 0, 3                    #
    # GROUP 2 : 10, 11, 12, 13, 14, 15  #
    # GROUP 3 : 8, 9                    #
    #####################################
    
    target_actor_lists = [
        [1, 2, 4, 5, 6, 7],
        [0, 3],
        [10, 11, 12, 13, 14, 15],
        [8, 9]
    ]
    
    Controller.setTrafficLightGroupGreen(target_actor_lists[0])

if __name__ == '__main__':
    main()