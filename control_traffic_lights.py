import traci
import random
import sumolib
import threading
import tkinter as tk
from tkinter import scrolledtext

junction_id = "6" # 실제 교차로 ID로 변경해야 함
stop_simulation = False # 전역 변수를 사용하여 시뮬레이션 중지 신호


# SUMO 실행 파일 경로 및 설정 파일 경로
sumoBinary = sumolib.checkBinary('sumo-gui')  # sumo 실행 파일 경로 설정
sumoCmd = [sumoBinary, "-c", "cross.sumocfg"]  # sumo 설정 파일 경로 설정


def generateRandomRoutes1():
    
    routes = [
    "L16 L10", "L9 -E0", "L12 -E0", "E0 L15", "E0 L10", 
    "L9 L11", "L9 L15", "E0 L11", "L12 L15", "L16 -E0", 
    "L16 L11", "L12 L10"
    ]

    vehicle_types = ["CarA", "CarB", "CarC", "CarD", "bus", "passenger", "taxi", "police", "emergency", "rail", "truck", "delivery", "passenger/hatchback", "passenger/sedan", "passenger/wagon", "passenger/van"]

    probability = 0.03 / len(vehicle_types)  # 분할된 확률 
    # 밀도 조정: 이 확률 값은 시뮬레이션의 차량 밀도를 조정하는 데 사용. 값이 클수록 시뮬레이션에 더 많은 차량이 생성되고, 작을수록 차량이 적게 생성

    with open("random_routes.rou.xml", "w") as f:
        f.write('<routes>\n')
        
        # Define vehicle types
        f.write('    <!-- vehicle types -->\n')
        f.write('    <vType accel="3.0" decel="6.0" id="CarA" length="5.0" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="20,20,20" />\n')
        f.write('    <vType accel="2.0" decel="6.0" id="CarB" length="4.5" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="255,255,255" />\n')
        f.write('    <vType accel="1.0" decel="5.0" id="CarC" length="5.0" minGap="2.5" maxSpeed="40.0" sigma="0.5" color="128,0,0" />\n')
        f.write('    <vType accel="1.0" decel="5.0" id="CarD" length="6.0" minGap="2.5" maxSpeed="30.0" sigma="0.5" color="0,128,0" />\n')
        f.write('    <vType accel="2.0" decel="5.0" id="bus" guiShape="bus" length="11.0" minGap="2.5" maxSpeed="40.0" sigma="0.5" color="0,128,128" />\n')
        f.write('    <vType accel="3.0" decel="6.0" id="passenger" guiShape="passenger" minGap="2.5" maxSpeed="50.0" color="0,255,0" sigma="0.5" />\n')
        f.write('    <vType accel="3.0" decel="6.0" id="taxi" guiShape="taxi" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="255,165,0" />\n')
        f.write('    <vType accel="4.0" decel="7.0" id="police" guiShape="police" minGap="2.5" maxSpeed="60.0" sigma="0.5" />\n')  # color="255,255,0" 
        f.write('    <vType accel="2.0" decel="6.0" id="emergency" guiShape="emergency" minGap="2.5" maxSpeed="60.0" sigma="0.5" />\n')  # color="128,128,255"
        f.write('    <vType accel="1.0" decel="4.0" id="rail" guiShape="rail" minGap="2.5" maxSpeed="30.0" sigma="0.5" color="255,0,128" />\n')
        f.write('    <vType accel="2.0" decel="4.0" id="truck" guiShape="truck" minGap="2.5" maxSpeed="40.0" sigma="0.5" color="0,255,255" />\n')
        f.write('    <vType accel="3.0" decel="6.0" id="delivery" guiShape="delivery" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="255,128,128" />\n')
        f.write('    <vType accel="3.0" decel="6.0" id="passenger/hatchback" guiShape="passenger/hatchback" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="144,238,144" />\n')
        f.write('    <vType accel="3.0" decel="6.0" id="passenger/sedan" guiShape="passenger/sedan" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="218,165,32" />\n')
        f.write('    <vType accel="3.0" decel="5.0" id="passenger/wagon" guiShape="passenger/wagon" minGap="2.5" maxSpeed="55.0" sigma="0.5" color="255,192,203" />\n')
        f.write('    <vType accel="4.0" decel="6.0" id="passenger/van" guiShape="passenger/van" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="173,216,230" />\n\n')

        
        # Define routes
        f.write('    <!-- Routes -->\n')
        for i, route in enumerate(routes):
            f.write(f'    <route id="r_{i}" edges="{route}"/>\n')
        f.write('\n')
        
        # Define flows
        f.write('    <!-- flows -->\n')
        flow_id = 0
        for i, route in enumerate(routes):
            for vehicle_type in vehicle_types:
                f.write(f'    <flow id="randomFlow{flow_id}" type="{vehicle_type}" begin="0" end="3600" probability="{probability}" route="r_{i}"/>\n')  
                # Probability: 각 flow가 특정 시간 간격마다 차량을 생성할 확률. 예를 들어, probability="0.01"은 매 초마다 1%의 확률로 차량이 생성됨을 의미
                flow_id += 1
                
        f.write('</routes>')
        
        
def generateRandomRoutes2():

    routes = [
        "L16 L10", "L9 -E0", "L12 -E0", "E0 L15", "E0 L10", 
        "L9 L11", "L9 L15", "E0 L11", "L12 L15", "L16 -E0", 
        "L16 L11", "L12 L10"
    ]

    vehicle_types = ["CarA", "CarB", "CarC", "CarD", "bus", "passenger", "taxi", "police", "emergency", "rail", "truck", "delivery", "passenger/hatchback", "passenger/sedan", "passenger/wagon", "passenger/van"]

    min_vehicles_per_route = 50  # 각 루트별 최소 차량 수
    max_vehicles_per_route = 150  # 각 루트별 최대 차량 수     --> 이 두 값을 올리면 전체적으로 시뮬레이션에 더 많은 차량이 생성될 확률이 발생

    vehicles = []

    # Define vehicle types
    vehicle_types_definition = '''
        <!-- vehicle types -->
        <vType accel="3.0" decel="6.0" id="CarA" length="5.0" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="20,20,20" />
        <vType accel="2.0" decel="6.0" id="CarB" length="4.5" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="255,255,255" />
        <vType accel="1.0" decel="5.0" id="CarC" length="5.0" minGap="2.5" maxSpeed="40.0" sigma="0.5" color="128,0,0" />
        <vType accel="1.0" decel="5.0" id="CarD" length="6.0" minGap="2.5" maxSpeed="30.0" sigma="0.5" color="0,128,0" />
        <vType accel="2.0" decel="5.0" id="bus" guiShape="bus" length="11.0" minGap="2.5" maxSpeed="40.0" sigma="0.5" color="0,128,128" />
        <vType accel="3.0" decel="6.0" id="passenger" guiShape="passenger" minGap="2.5" maxSpeed="50.0" color="0,255,0" sigma="0.5" />
        <vType accel="3.0" decel="6.0" id="taxi" guiShape="taxi" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="255,165,0" />
        <vType accel="4.0" decel="7.0" id="police" guiShape="police" minGap="2.5" maxSpeed="60.0" sigma="0.5" /> 
        <vType accel="2.0" decel="6.0" id="emergency" guiShape="emergency" minGap="2.5" maxSpeed="60.0" sigma="0.5" />
        <vType accel="1.0" decel="4.0" id="rail" guiShape="rail" minGap="2.5" maxSpeed="30.0" sigma="0.5" color="255,0,128" />
        <vType accel="2.0" decel="4.0" id="truck" guiShape="truck" minGap="2.5" maxSpeed="40.0" sigma="0.5" color="0,255,255" />
        <vType accel="3.0" decel="6.0" id="delivery" guiShape="delivery" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="255,128,128" />
        <vType accel="3.0" decel="6.0" id="passenger/hatchback" guiShape="passenger/hatchback" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="144,238,144" />
        <vType accel="3.0" decel="6.0" id="passenger/sedan" guiShape="passenger/sedan" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="218,165,32" />
        <vType accel="3.0" decel="5.0" id="passenger/wagon" guiShape="passenger/wagon" minGap="2.5" maxSpeed="55.0" sigma="0.5" color="255,192,203" />
        <vType accel="4.0" decel="6.0" id="passenger/van" guiShape="passenger/van" minGap="2.5" maxSpeed="50.0" sigma="0.5" color="173,216,230" />
    '''

    # Define routes
    routes_definition = '<routes>\n'
    routes_definition += '    <!-- Routes -->\n'
    for i, route in enumerate(routes):
        routes_definition += f'    <route id="r_{i}" edges="{route}"/>\n'
    routes_definition += '\n'

    # Generate vehicles
    vehicle_id = 0
    for i, route in enumerate(routes):
        total_vehicles = random.randint(min_vehicles_per_route, max_vehicles_per_route)
        for _ in range(total_vehicles):
            vehicle_type = random.choice(vehicle_types)
            depart_time = random.uniform(0, 3600)  # 0초부터 3600초 사이의 임의 시간에 출발
            vehicles.append((depart_time, f'    <vehicle id="veh{vehicle_id}" type="{vehicle_type}" route="r_{i}" depart="{depart_time:.2f}" />\n'))
            vehicle_id += 1

    # Sort vehicles by departure time
    vehicles.sort(key=lambda x: x[0])

    # Write to XML file
    with open("random_routes.rou.xml", "w") as f:
        f.write('<routes>\n')
        f.write(vehicle_types_definition)
        f.write(routes_definition)
        f.write('    <!-- vehicles -->\n')
        for _, vehicle_definition in vehicles:
            f.write(vehicle_definition)
        f.write('</routes>')


def getInwardLanes(junction_id):
    inward_lanes = []
    # Get the edges leading to the junction
    
    # edge_ids = traci.junction.getIncomingEdges(junction_id)

    # for edge_id in edge_ids:        
    #     lanes = traci.edge.getLaneNumber(edge_id)
    #     for i in range(lanes):
    #         lane_id = f"{edge_id}_{i}"
    #         inward_lanes.append(lane_id)
    
    # 제가 만든 network(cross.net.xml)에 의거, junction에 진입하는 차로(lane_id)들의 목록입니다 (이 차로들만 보시면 됨)
    inward_lanes = ["E0_0", "E0_1", "E0_2", "L9_0", "L9_1", "L9_2", "L16_0", "L16_1", "L16_2", "L12_0", "L12_1", "L12_2"]
            
    return inward_lanes




# getEachLaneWaitingStats 함수 설명:
# 각 차로(lane)의 대기시간(평균/최대) 및 대기차량수를 traCI 인터페이스로 추출
# 리턴값 : 각 lane별 차량대기 정보를 나타내는, 아래와 같은 형태의 dictionary 변수를 원소로 가지는 리스트를 반환합니다
        # {
        #     'lane_id': lane_id,  #
        #     'average_waiting_time': average_waiting_time,
        #     'max_waiting_time': max_waiting_time,
        #     'vehicle_count': vehicle_count
        # }
def getEachLaneWaitingStats(junction_id):
    ## returns dictionary array, of which index and dictionary, corresponds to the lane Number(assign 0~11) and the wating information of each lane ##
    ## total number of lanes: number of every lanes that are heading inward to junction(this case: 12)
        
    lane_ids = getInwardLanes(junction_id)
    
    waiting_stats = []    
    
    for lane_id in lane_ids:
        total_waiting_time = 0
        vehicle_count = 0
        
        # 차로별 대기차량 수를 카운트(속도가 0.1미만인 차량들만)
        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
        for vehicle in vehicles:
            waiting_time = traci.vehicle.getWaitingTime(vehicle)
            total_waiting_time += waiting_time
            speed = traci.vehicle.getSpeed(vehicle)
            if speed < 0.1:  # 차량이 멈춘 상태로 간주 (속도가 0.1 미만)
                vehicle_count += 1
        
        average_waiting_time = total_waiting_time / vehicle_count if vehicle_count > 0 else 0
        max_waiting_time = max([traci.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]) if vehicles else 0
        
        waiting_stats.append({
            'lane_id': lane_id,  # 차로 id
            'average_waiting_time': average_waiting_time, # 해당 차로 평균 차량 대기시간
            'max_waiting_time': max_waiting_time, # 해당 차로 최대 차량 대기시간
            'vehicle_count': vehicle_count  # 해당 차로 대기 차량 수
        })
    
    return waiting_stats


## TraCI로 추출된 각 차선별 신호대기 통계를 출력하는 부분
def printWaitingStats(junction_id):
    waiting_stats = getEachLaneWaitingStats(junction_id)
    output = "\n===== Traffic Waiting Statistics  =====\n"
    for lane_data in waiting_stats:
        output += f"Lane {lane_data['lane_id']}:\n"
        output += f"  Avg. Waiting Time: {lane_data['average_waiting_time']} seconds\n"
        output += f"  Max. Waiting Time: {lane_data['max_waiting_time']} seconds\n"
        output += f"  Waiting Car Count: {lane_data['vehicle_count']}\n"
    output += "===============================================\n"
    return output




# TraCI 인터페이스를 사용하여 진행중인 시뮬레이션의 신호등 신호길이를 실시간으로 변경하는 함수
# tlsID : 신호등 ID,  phaseIndex : 해당 신호등의 신호 phase 중 하나(e.g. 0~7 중 하나), duration : 해당 신호 phase에 적용할 새로운 신호길이(sec)
def setSig(tlsID, phaseIndex, duration):
    ## define function that sets traffic signal of traffic light with id 'tlsID' to given parameters ##
    
    # tlsID = traci.trafficlight.getIDList()[0]  # 첫 번째 신호등(현 케이스: 교차로에 존재하는 유일한 신호등)의 ID를 가져옴
    
    program = traci.trafficlight.getAllProgramLogics(tlsID)
    logic = program[0]    # 신호등의 현재 논리(제어 방식)를 가져옴
    
    # 'phaseIndex' 단계의 지속 시간을 duration초로 변경하고 minDur와 maxDur 설정
    logic.phases[phaseIndex].duration = duration   # 지속시간을 duration으로 변경
    logic.phases[phaseIndex].minDur = duration
    logic.phases[phaseIndex].maxDur = duration
    print(logic.getPhases())
    
    # 변경된 프로그램을 설정합니다.
    traci.trafficlight.setProgramLogic(tlsID, logic)  # 변경된 논리를 다시 설정함
    return





#############################################################################################################################
## 아래 코드들: 명령어(콘솔 입력으로 신호길이 변경)를 입력받는 쓰레드와 시뮬레이션을 제어(및 출력)하는 쓰레드 두개로 나눔  ##
## 추가적으로 양 쓰레드의 입출력 부분이 명확하게 나뉘도록, Tk GUI 툴킷의 파이썬 바인딩인 tkinter를 사용                    ##
#############################################################################################################################



# 시뮬레이션 제어(및 대기시간 통계 출력) 스레드
def simulation_control(output_widget):
    global stop_simulation
    step = 0
    junction_id = "6"  # 실제 교차로 ID로 변경해야 함
    tlsID = traci.trafficlight.getIDList()[0]  # 첫 번째 신호등의 ID를 가져옴
 
    # 시뮬레이션 반복
    step = 0
    while step < 3600 and not stop_simulation:
        traci.simulationStep()  # 시뮬레이션 한 단계를 진행함
        
        if step % 1 == 0 : # 매초(1스텝) 마다 
            output = printWaitingStats(junction_id) # 각 차선별 신호대기 통계를 출력
            output_widget.insert(tk.END, output)
            output_widget.see(tk.END)
            
        step += 1  # 단계를 증가시킴   
    
    # 시뮬레이션 종료   
    traci.close()  # sumo 연결을 종료함


# 명령어(콘솔 입력으로 신호길이 변경)를 입력받는 쓰레드
def on_command_entry(event, input_widget, output_widget):
    global stop_simulation
    tlsID = traci.trafficlight.getIDList()[0]  # 첫 번째 신호등의 ID를 가져옴(어차피 이 network에서는 교차로가 하나라, 신호등 id도 이것 하나뿐입니다)

    command = input_widget.get()
    if command == "exit":
        stop_simulation = True
    else:
        try:
            # 명령 형식: setSig <phaseIndex> <duration>
            # 예시 : setSig 2 40
            #        (2번 신호 phase의 신호길이를 40초로 변경)
            parts = command.split()
            phaseIndex = int(parts[1])
            duration = int(parts[2])
            setSig(tlsID, phaseIndex, duration)  # TraCI를 이용해 시뮬레이션의 특정 신호의 길이를 변화시키는 함수
        except Exception as e:
            output_widget.insert(tk.END, f"Invalid command or error occurred: {e}\n")
            output_widget.see(tk.END)
    
    input_widget.delete(0, tk.END)   
            


def main():
    global stop_simulation
    
    # generateRandomRoutes1()     # 방법1. 각 route에 랜덤한 flow를 설정하는 방식으로 rou.xml 파일 생성
    generateRandomRoutes2()   # 방법2. 시뮬레이션에서 생성할 모든 개별 vehicle를 랜덤하게 정의하는 방식으로 rou.xml 파일 생성
    
    # 시뮬레이션 시작
    traci.start(sumoCmd)  # sumo를 실행하고 연결을 시작함

    # GUI setup
    root = tk.Tk()
    root.title("SUMO Simulation Control")

    output_frame = tk.Frame(root)
    output_frame.pack(fill=tk.BOTH, expand=True)
    output_widget = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD)
    output_widget.pack(fill=tk.BOTH, expand=True)

    input_frame = tk.Frame(root)
    input_frame.pack(fill=tk.X)
    input_widget = tk.Entry(input_frame)
    input_widget.pack(fill=tk.X, expand=True)
    input_widget.bind("<Return>", lambda event: on_command_entry(event, input_widget, output_widget))

    def run_simulation():
        simulation_control(output_widget)
        root.quit()

    # Start simulation control thread
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.start()

    root.mainloop()

    # Wait for the simulation thread to finish
    simulation_thread.join()

    print("Simulation ended.")

if __name__ == "__main__":
    main()



 # # 시뮬레이션 중간에 신호등 제어 변경
    # if step == 100:
    #     print(f"Step {step}:")
    #     printWaitingStats(junction_id)
    #     setSig(0, 45)
    # if step == 200:
    #     print(f"Step {step}:")
    #     printWaitingStats(junction_id)
    #     setSig(1, 20)
    # if step == 300:
    #     print(f"Step {step}:")
    #     printWaitingStats(junction_id)
    #     setSig(0, 35)
    # if step == 400:
    #     print(f"Step {step}:")
    #     printWaitingStats(junction_id)
    #     setSig(2, 50)
    # if step == 500:
    #     print(f"Step {step}:")
    #     printWaitingStats(junction_id)
    #     setSig(4, 15)




# def getEachLaneWaitingStats(junction_id):
    ## returns int array, of which index and value, corresponds to the lane Number(assign 0~11) and the wating time of the first car waiting in each lane ##
    ## total number of lanes: number of every lanes that are heading inward to junction(this case: 12)
   
   
    # lane_ids = getInwardLanes(junction_id)
    
    # waiting_times = []
    
    # for lane_id in lane_ids:
    #     vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
    #     if vehicles:
    #         # 첫 번째 차량의 대기 시간을 가져옵니다.
    #         first_vehicle = vehicles[0]
            
    #         waiting_time = traci.vehicle.getWaitingTime(first_vehicle)
    #             # getWaitingTime :
    #             # The waiting time of a vehicle is defined as the time (in seconds) spent with a
    #             # speed below 0.1m/s since the last time it was faster than 0.1m/s.
    #             # (basically, the waiting time of a vehicle is reset to 0 every time it moves).
    #             # A vehicle that is stopping intentionally with a <stop> does not accumulate waiting time.
                
    #         # waiting_time = traci.vehicle.getAccumulatedWaitingTime(first_vehicle)
    #             # getAccumulatedWaitingTime :   
    #             # Returns the accumulated waiting time [s] within the previous time interval of default length 100 s. 
    #             # (length is configurable per option --waiting-time-memory given to the main application)
    #     else:
    #         waiting_time = 999.999
    #     waiting_times.append(waiting_time)
    
    # return waiting_stats