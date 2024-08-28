from control_traffic_lights import calculateReward, generateRandomRoutes2, getEachLaneWaitingStats, printWaitingStats, setSig
import traci
import os
import sumolib
import numpy as np 
from control_traffic_lights import stop_simulation


class SumoEnv:
    def __init__(self, sumo_config, sumo_binary='sumo-gui'):
        self.sumo_config = sumo_config  # "cross.sumocfg"
        self.sumo_binary = sumolib.checkBinary(sumo_binary)  # sumo 실행 파일 경로 설정
        self.sumo_process = None

    def start(self):
        generateRandomRoutes2() 
        
        if self.sumo_process is None:
            sumo_cmd = [
                self.sumo_binary, 
                "-c", self.sumo_config,
                "--start",
                "--step-length", "10"
            ]
            self.sumo_process = traci.start(sumo_cmd)

    def reset(self):
        if self.sumo_process is not None:
            generateRandomRoutes2() 
            traci.load(["-c", self.sumo_config, "--start", "--step-length", "1"])
        else:
            self.start()
            
        next_state = self.get_state()
        next_state = np.append(next_state, self.get_state())
        next_state = np.append(next_state, self.get_state())
        next_state = np.append(next_state, self.get_state())
        next_state = np.append(next_state, self.get_state())
        
        return next_state

    def step(self, action):
        # Implement the logic to apply the action and advance the simulation
        
        tlsID = traci.trafficlight.getIDList()[0]  # 첫 번째 신호등의 ID를 가져옴(어차피 이 network에서는 교차로가 하나라, 신호등 id도 이것 하나뿐입니다)
        program = traci.trafficlight.getAllProgramLogics(tlsID)
        logic = program[0]    # 신호등의 현재 논리(제어 방식)를 가져옴
        
        if action == 0:
            duration = logic.phases[0].duration + 0.5 if logic.phases[0].duration < 25 else logic.phases[0].duration
            setSig(tlsID, 0, duration)
        elif action == 1:
            duration = logic.phases[2].duration + 0.5 if logic.phases[2].duration < 25 else logic.phases[2].duration
            setSig(tlsID, 2, duration)
        elif action == 2:
            duration = logic.phases[4].duration + 0.5 if logic.phases[4].duration < 25 else logic.phases[4].duration
            setSig(tlsID, 4, duration)       
        elif action == 3:
            duration = logic.phases[6].duration + 0.5 if logic.phases[6].duration < 25 else logic.phases[6].duration
            setSig(tlsID, 6, duration)      
        elif action == 4:
            duration = logic.phases[0].duration - 0.5 if logic.phases[0].duration > 3 else logic.phases[0].duration
            setSig(tlsID, 0, duration)       
        elif action == 5:
            duration = logic.phases[2].duration - 0.5 if logic.phases[2].duration > 3 else logic.phases[2].duration
            setSig(tlsID, 2, duration)       
        elif action == 6:
            duration = logic.phases[4].duration - 0.5 if logic.phases[4].duration > 3 else logic.phases[4].duration
            setSig(tlsID, 4, duration)       
        elif action == 7:
            duration = logic.phases[6].duration - 0.5 if logic.phases[6].duration > 3 else logic.phases[6].duration
            setSig(tlsID, 6, duration)    
        
        print(logic.phases[0].duration, logic.phases[1].duration, logic.phases[2].duration, logic.phases[3].duration, logic.phases[4].duration, logic.phases[5].duration, logic.phases[6].duration, logic.phases[7].duration) 
        
        
        traci.simulationStep()
        next_state = self.get_state()
        
        traci.simulationStep()
        traci.simulationStep()
        next_state = np.append(next_state, self.get_state())

        traci.simulationStep()
        traci.simulationStep()
        next_state = np.append(next_state, self.get_state())

        traci.simulationStep()
        traci.simulationStep()
        next_state = np.append(next_state, self.get_state())

        traci.simulationStep()
        traci.simulationStep()
        next_state = np.append(next_state, self.get_state())
        
        reward = self.get_reward()
        done = self.is_done()
        return next_state, reward, done, {}


    def get_state(self):
        # Implement the logic to get the current state of the environment

        junction_id = "6"  # 교차로 ID
        output = getEachLaneWaitingStats(junction_id) 
        # 다음과 같은 배열 리턴 
        # [
        #   [ 현재신호주기, 다음신호남은시간, 현신호지난시간 ], 
        #   [ 신호주기0길이, 신호주기1길이, 신호주기2길이, 신호주기3길이, 신호주기4길이, 신호주기5길이, 신호주기6길이, 신호주기7길이 ], 
        #   [ 0, lane0 전체 차량수, 정지 차량수, 평균 속도, 평균 가속도 ], ...(및11개 lane)..., 
        # ]
        
        state = np.array(output)
        
        return state

    def get_reward(self):
        # Implement the logic to calculate the reward
        return calculateReward()
        
        
    def is_done(self):
        # Implement the logic to check if the episode is done
        global stop_simulation
        step = traci.simulation.getTime()
        
        if step <= 600 and not stop_simulation:
            return False
        else:
            return True
        

    def close(self):
        if self.sumo_process is not None:
            traci.close()
            self.sumo_process = None

def make(sumo_config, sumo_binary="sumo-gui"):
    return SumoEnv(sumo_config, sumo_binary)