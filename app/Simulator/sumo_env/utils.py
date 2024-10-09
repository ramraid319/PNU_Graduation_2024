import io
import tempfile
import traci
import random
import sumolib
import threading
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import os
from PIL import Image
from OpenGL import GL
from io import BytesIO
import mss

junction_id = "6" # 실제 교차로 ID로 변경해야 함
stop_simulation = False # 전역 변수를 사용하여 시뮬레이션 중지 신호
i = 0
flag = 0


# SUMO 실행 파일 경로 및 설정 파일 경로
sumoBinary = sumolib.checkBinary('sumo-gui')  # sumo 실행 파일 경로 설정
sumoCmd = [sumoBinary, "-c", "cross.sumocfg"]  # sumo 설정 파일 경로 설정


 # 방법1. 각 route에 랜덤한 flow를 설정하는 방식으로 rou.xml 파일 생성
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
                f.write(f'    <flow id="randomFlow{flow_id}" type="{vehicle_type}" begin="0" end="360" probability="{probability}" route="r_{i}"/>\n')  
                # Probability: 각 flow가 특정 시간 간격마다 차량을 생성할 확률. 예를 들어, probability="0.01"은 매 초마다 1%의 확률로 차량이 생성됨을 의미
                flow_id += 1
                
        f.write('</routes>')


        
# 방법2. 시뮬레이션에서 생성할 모든 개별 vehicle를 랜덤하게 정의하는 방식으로 rou.xml 파일 생성   ( <== 지금은 방법2를 사용합니다!! )     
def generateRandomRoutes2():

    routes = [
        "L16 L10", "L9 -E0", "L12 -E0", "E0 L15", "E0 L10", 
        "L9 L11", "L9 L15", "E0 L11", "L12 L15", "L16 -E0", 
        "L16 L11", "L12 L10"
    ] 
    # start : L16, L9, L12, E0 
    # end : L10, -E0, L15, L11
    
    ped_routes = [
        "L16 L9", "L16 L12", "L16 E0", "L16 L10", "L16 -E0", "L16 L15", "L16 L11",
        "L9 L16", "L9 L12", "L9 E0", "L9 L10", "L9 -E0", "L9 L15", "L9 L11",
        "L12 L16", "L12 L9", "L12 E0", "L12 L10", "L12 -E0", "L12 L15", "L12 L11",
        "E0 L16", "E0 L9", "E0 L12", "E0 L10", "E0 -E0", "E0 L15", "E0 L11",
        "L10 L16", "L10 L9", "L10 L12", "L10 E0", "L10 -E0", "L10 L15", "L10 L11",
        "-E0 L16", "-E0 L9", "-E0 L12", "-E0 E0", "-E0 L10", "-E0 L15", "-E0 L11",
        "L15 L16", "L15 L9", "L15 L12", "L15 E0", "L15 L10", "L15 -E0", "L15 L11",
        "L11 L16", "L11 L9", "L11 L12", "L11 E0", "L11 L10", "L11 -E0", "L11 L15"
    ]   
    # start & end : L16, L9, L12, E0, L10, -E0, L15, L11

    vehicle_types = ["CarA", "CarB", "CarC", "CarD", "bus", "passenger", "taxi", "police", "emergency", "rail", "truck", "delivery", "passenger/hatchback", "passenger/sedan", "passenger/wagon", "passenger/van"]

    ped_vehicle_types = ["ped0", "ped1", "ped2", "ped3", "ped4", "ped5"]

    min_vehicles_per_route = 110   #25    #50   각 루트별 최소 차량 수
    max_vehicles_per_route = 130   #75    #150   각 루트별 최대 차량 수     --> 이 두 값을 올리면 전체적으로 시뮬레이션에 더 많은 차량이 생성될 확률이 발생

    min_pedestrian_per_route = 0   
    max_pedestrian_per_route = 0 


    vehicles = []

    # Define vehicle types
    vehicle_types_definition = '''
        <!-- vehicle types (changes made : accelx2, decelx2, maxSpeedx2) -->
        <vType accel="6.0" decel="12.0" id="CarA" length="5.0" minGap="2.5" maxSpeed="100.0" sigma="0.5" color="20,20,20" /> 
        <vType accel="4.0" decel="12.0" id="CarB" length="4.5" minGap="2.5" maxSpeed="100.0" sigma="0.5" color="255,255,255" />
        <vType accel="2.0" decel="10.0" id="CarC" length="5.0" minGap="2.5" maxSpeed="80.0" sigma="0.5" color="128,0,0" />
        <vType accel="2.0" decel="10.0" id="CarD" length="6.0" minGap="2.5" maxSpeed="60.0" sigma="0.5" color="0,128,0" />
        <vType accel="4.0" decel="10.0" id="bus" guiShape="bus" length="11.0" minGap="2.5" maxSpeed="80.0" sigma="0.5" color="0,128,128" />
        <vType accel="6.0" decel="12.0" id="passenger" guiShape="passenger" minGap="2.5" maxSpeed="100.0" color="0,255,0" sigma="0.5" />
        <vType accel="6.0" decel="12.0" id="taxi" guiShape="taxi" minGap="2.5" maxSpeed="100.0" sigma="0.5" color="255,165,0" />
        <vType accel="8.0" decel="14.0" id="police" guiShape="police" minGap="2.5" maxSpeed="120.0" sigma="0.5" /> 
        <vType accel="4.0" decel="12.0" id="emergency" guiShape="emergency" minGap="2.5" maxSpeed="120.0" sigma="0.5" />
        <vType accel="2.0" decel="8.0" id="rail" guiShape="rail" minGap="2.5" maxSpeed="60.0" sigma="0.5" color="255,0,128" />
        <vType accel="4.0" decel="8.0" id="truck" guiShape="truck" minGap="2.5" maxSpeed="80.0" sigma="0.5" color="0,255,255" />
        <vType accel="6.0" decel="12.0" id="delivery" guiShape="delivery" minGap="2.5" maxSpeed="100.0" sigma="0.5" color="255,128,128" />
        <vType accel="6.0" decel="12.0" id="passenger/hatchback" guiShape="passenger/hatchback" minGap="2.5" maxSpeed="100.0" sigma="0.5" color="144,238,144" />
        <vType accel="6.0" decel="12.0" id="passenger/sedan" guiShape="passenger/sedan" minGap="2.5" maxSpeed="100.0" sigma="0.5" color="218,165,32" />
        <vType accel="6.0" decel="10.0" id="passenger/wagon" guiShape="passenger/wagon" minGap="2.5" maxSpeed="110.0" sigma="0.5" color="255,192,203" />
        <vType accel="8.0" decel="12.0" id="passenger/van" guiShape="passenger/van" minGap="2.5" maxSpeed="100.0" sigma="0.5" color="173,216,230" />           
        <vType vClass="pedestrian" id="ped0" guiShape="pedestrian" speedFactor="0.7" width="0.66" length="0.36" height="1.65" color="255,0,128" />
        <vType vClass="pedestrian" id="ped1" guiShape="pedestrian" speedFactor="0.7" width="0.59" length="0.38" height="1.73" color="255,165,0" />
        <vType vClass="pedestrian" id="ped2" guiShape="pedestrian" speedFactor="1.0" width="0.55" length="0.34" height="1.60" color="255,128,128" />
        <vType vClass="pedestrian" id="ped3" guiShape="pedestrian" speedFactor="1.0" width="0.60" length="0.40" height="1.75" color="144,238,144" />
        <vType vClass="pedestrian" id="ped4" guiShape="pedestrian" speedFactor="1.3" width="0.58" length="0.37" height="1.68" color="128,128,255" />
        <vType vClass="pedestrian" id="ped5" guiShape="pedestrian" speedFactor="1.3" width="0.57" length="0.39" height="1.70" color="100,87,240" />
    '''

    # Define routes
    # routes_definition = '<routes>\n'
    routes_definition = '    <!-- Routes -->\n'
    for i, route in enumerate(routes):
        routes_definition += f'    <route id="r_{i}" edges="{route}"/>\n'
    routes_definition += '\n'
    
    # Define pedestrian routes
    routes_definition += '    <!-- Pedestrian Routes -->\n'
    for i, ped_route in enumerate(ped_routes):
        routes_definition += f'    <route id="ped_r_{i}" edges="{ped_route}"/>\n'
    routes_definition += '\n'

    # Generate vehicles(cars)
    vehicle_id = 0
    for j, route in enumerate(routes):
        # if j == 5 or j == 11:
        #     total_vehicles = random.randint(80, 120)
        # else:
        #     total_vehicles = random.randint(min_vehicles_per_route, max_vehicles_per_route)
        total_vehicles = random.randint(min_vehicles_per_route, max_vehicles_per_route)
        for _ in range(total_vehicles):
            vehicle_type = random.choice(vehicle_types)
            depart_time = random.uniform(0, 2000)  # 0초부터 900초 사이의 임의 시간에 출발
            vehicles.append((depart_time, f'    <vehicle id="veh{vehicle_id}" type="{vehicle_type}" route="r_{j}" depart="{depart_time:.2f}" />\n'))
            vehicle_id += 1
    # Generate vehicles(pedestrians)
    for j, route in enumerate(ped_routes):
        total_vehicles = random.randint(min_pedestrian_per_route, max_pedestrian_per_route)
        for _ in range(total_vehicles):
            vehicle_type = random.choice(ped_vehicle_types)
            depart_time = random.uniform(0, 2000)  # 0초부터 3600초 사이의 임의 시간에 출발
            vehicles.append((depart_time, f'    <person id="veh{vehicle_id}" type="{vehicle_type}" depart="{depart_time:.2f}" departPos="0"><walk route="ped_r_{j}" arrivalPos="87.15" departPosLat="random"/></person>\n'))    
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
    inward_lanes = ["E0_1", "E0_2", "E0_3", "L9_1", "L9_2", "L9_3", "L16_1", "L16_2", "L16_3", "L12_1", "L12_2", "L12_3"]
            
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
def getSimStatebyNumbers(junction_id):
    ## returns dictionary array, of which index and dictionary, corresponds to the lane Number(assign 0~11) and the wating information of each lane ##
    ## total number of lanes: number of every lanes that are heading inward to junction(this case: 12)

    lane_ids = getInwardLanes(junction_id)
    
    waiting_stats = []    

    tlsID = traci.trafficlight.getIDList()[0]  # 첫 번째 신호등의 ID를 가져옴(어차피 이 network에서는 교차로가 하나라, 신호등 id도 이것 하나뿐입니다)
    current_phase = traci.trafficlight.getPhase(tlsID)
    assumed_time_next_switch = traci.trafficlight.getNextSwitch(tlsID) - traci.simulation.getTime()

    spent_duration = traci.trafficlight.getSpentDuration(tlsID)

    waiting_stats.extend([current_phase, spent_duration])    
    
    
    program = traci.trafficlight.getAllProgramLogics(tlsID)
    logic = program[0]    # 신호등의 현재 논리(제어 방식)를 가져옴
    
    phases = logic.getPhases()
    durations = [phase.duration for phase in phases]   # 현재시점 각 phase별 duration들만 들어간 배열
    
    ## waiting_stats.extend(durations)  

    laneNo = 0
    for lane_id in lane_ids:
        total_waiting_time = 0
        waiting_vehicle_count = 0
        total_vehicle_count_on_lane = traci.lane.getLastStepVehicleNumber(lane_id)
        sum_speed = 0
        sum_acceleration = 0

        
        # 차로별 대기차량 수를 카운트(속도가 0.1미만인 차량들만)
        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)  
        for vehicle in vehicles:
            sum_speed += traci.vehicle.getSpeed(vehicle)
            sum_acceleration += traci.vehicle.getAcceleration(vehicle)
            waiting_time = traci.vehicle.getWaitingTime(vehicle)
            total_waiting_time += waiting_time
            speed = traci.vehicle.getSpeed(vehicle)
            if speed < 0.1:  # 차량이 멈춘 상태로 간주 (속도가 0.1 미만)
                waiting_vehicle_count += 1
        
        avg_speed = sum_speed / total_vehicle_count_on_lane if total_vehicle_count_on_lane > 0 else 0
        avg_acceleration = sum_acceleration / total_vehicle_count_on_lane if total_vehicle_count_on_lane > 0 else 0
        
        
        # 평균 대기시간 구하기
        average_waiting_time = total_waiting_time / waiting_vehicle_count if waiting_vehicle_count > 0 else 0
        # 최장 대기시간 구하기
        max_waiting_time = max([traci.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]) if vehicles else 0
        # 차량 평균 속도
                
        waiting_stats.extend([
            # 'lane_id': lane_id,  # 차로 id
            # laneNo,  # 차로번호(여기선 임의로 0에서 카운트업)
            total_vehicle_count_on_lane,  # lane상의 전체 차량 수
            waiting_vehicle_count,  # lane상의 대기(정지) 차량 수
            round(avg_speed,4),   # lane상 모든 차량의 평균 속도
            # round(avg_acceleration,2) # lane상 모든 차량의 평균 가속도
            # 'average_waiting_time': average_waiting_time, # 해당 차로 평균 차량 대기시간
            # 'max_waiting_time': max_waiting_time, # 해당 차로 최대 차량 대기시간
        ])
        laneNo += 1
    
    return waiting_stats  # state배열 길이 : 2 + 12 * 3 = 2 + 36 = 38


# 시뮬레이션의 state를 숫자 배열이 아닌 스크린샷 이미지(2차원 배열)로 리턴합니다.
def getSimStatebyImage():    
    # Define the screenshot directory
    # screenshot_dir = os.path.join(os.getcwd(), "screenshots")  # Save in the current working directory

    # # Ensure the screenshot directory exists
    # if not os.path.exists(screenshot_dir):
    #     os.makedirs(screenshot_dir)

    # screenshot_path = os.path.join(screenshot_dir, "screenshot.png")
    # # screenshot_path = r"C:\Users\Georges Martel\Documents\부산대학교\4-1\졸업과제\SUMO-projects\control_cross_signal\control_cross_signal\screenshots\screenshot.png"
    
    # # 1. Capture screenshot from the SUMO GUI using TraCI
    # screenshot_buffer = traci.gui.screenshot(viewID=traci.gui.DEFAULT_VIEW, filename="")
    
    # print(screenshot_buffer)
    # image = Image.open(io.BytesIO(screenshot_buffer))
    
    # Capture the current frame from the SUMO GUI
    # image_array = grab_framebuffer()
    
    # image = Image.fromarray(image_array)

    # traci.simulationStep()
    # # Verify if the screenshot is saved
    # if not os.path.exists(screenshot_path):
    #     raise FileNotFoundError(f"Screenshot was not saved at {screenshot_path}")
    
    global flag
    
    # Initialize mss for screen capture
    with mss.mss() as sct:
        # Define the monitor region to capture (this needs to be adjusted to your SUMO window)
        # Example region (left, top, width, height). Adjust to your SUMO window size/position.
        monitor = {"top": 201, "left": 132, "width": 737, "height": 737}
        # 좌에서 132, 상에서 201  (전체화면 1920 * 1080 fhd 해상도 기준)
        # 869 - 132 = 737

        # Capture the screen
        screenshot = sct.grab(monitor)

        # Convert the screenshot to a NumPy array (RGB format)
        image_array = np.array(screenshot)
        
        if (flag == 100):
            image = Image.fromarray(image_array)
            image.save("sumo_screenshot_before_before.png")

        # The `mss` screenshot contains 4 channels (RGBA), so we can discard the alpha channel
        image_array = image_array[..., :3]  # Keep only the RGB channels
    
        # Convert the image to a PIL image to use its transformation functions
        image = Image.fromarray(image_array)

        if (flag == 100):
            # image = Image.fromarray(image_array)
            image.save("sumo_screenshot_before.png")

        # 3. Preprocess the image
        # Resize the image to match the input shape of the CNN
        image = image.resize((120, 120))
    
        # convert to grayscale to reduce input complexity
        image = image.convert('L')  # 'L' mode is for grayscale 
        
        
        if (flag == 100):
            # image = Image.fromarray(image_array)
            image.save("sumo_screenshot_after.png")
        flag += 1   

        # Convert the image back to a NumPy array
        image_array = np.array(image)
    
        # Normalize pixel values to be between 0 and 1
        image_array = image_array / 255.0
    
        # Add an extra dimension to match the input shape of the CNN
        # (80, 80, 1) if grayscale, or (80, 80, 3) for RGB images
        image_array = np.expand_dims(image_array, axis=-1)
    
        # 4. Return the preprocessed image
        return image_array   # (80, 80, 1) is returned!

def grab_framebuffer():
    # Define the size of the viewport (window size of SUMO GUI)
    # width, height = 600, 600  # Adjust this to match your SUMO window dimensions
    
    viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
    width, height = viewport[2], viewport[3]
    
    gl_pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
    
    # Check if there's an OpenGL error
    error = GL.glGetError()
    if error != GL.GL_NO_ERROR:
        raise RuntimeError(f"OpenGL error: {error}")
    
    # Convert the pixel data to a NumPy array
    image_array = np.frombuffer(gl_pixels, dtype=np.uint8).reshape(height, width, 3)
    
    # Flip the image vertically as OpenGL reads from bottom to top
    image_array = np.flipud(image_array)
    
    return image_array



# reward를 계산합니다
def calculateReward():
    
    lane_ids = getInwardLanes(junction_id)  
    sum_total_waiting_time = 0   
    sum_sum_speed = 0
    total_vehicle_count_on_juction = 0
    sum_waiting_vehicle_count = 0
    lane_max_waiting_times = []
    
    for lane_id in lane_ids:
        total_waiting_time = 0
        waiting_vehicle_count = 0
        total_vehicle_count_on_lane = traci.lane.getLastStepVehicleNumber(lane_id)
        sum_speed = 0
      
        # 차로별 대기차량 수를 카운트(속도가 0.1미만인 차량들만)
        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)  
        for vehicle in vehicles:
            sum_speed += traci.vehicle.getSpeed(vehicle)
            waiting_time = traci.vehicle.getWaitingTime(vehicle)
            total_waiting_time += waiting_time   ## = (1.0001) ** (waiting_time)
            speed = traci.vehicle.getSpeed(vehicle)
            if speed < 0.1:  # 차량이 멈춘 상태로 간주 (속도가 0.1 미만)
                waiting_vehicle_count += 1
                
        # 최장 대기시간 구하기
        max_waiting_time = max([traci.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]) if vehicles else 0
        lane_max_waiting_times.append(max_waiting_time)
        
        sum_total_waiting_time += total_waiting_time
        sum_sum_speed += sum_speed
        total_vehicle_count_on_juction += total_vehicle_count_on_lane
        sum_waiting_vehicle_count += waiting_vehicle_count

    
    universal_max_waiting_time = max(lane_max_waiting_times)
    total_average_waiting_time = sum_total_waiting_time / sum_waiting_vehicle_count if sum_waiting_vehicle_count > 0 else 0
    
    
    # 작을수록 좋은 지표 => y = 100 * ( (9994/10000) ** x )
    
    # reward = 100 * ( (9994/10000) ** sum_total_waiting_time )   # 모든 대기(정지)차량의 대기시간 총합
    reward = -1 * total_average_waiting_time   # 모든 대기차량의 평균 대기시간
    # reward = universal_max_waiting_time   # 가장 오래 기다린 차량의 대기시간
    # reward = total_vehicle_count_on_juction   # 교차로 진입하는 도로 상의 모든 차량의 수
    # reward = sum_waiting_vehicle_count    # 대기하는 차량의 총 수
    
    
    # 클수록 좋은 지표 =>  y= -1 * 100 * ((9994/10000) ** x) + 100
    # reward = sum_sum_speed    #
    
    
    # reward = -1 * sum_total_waiting_time   # - ( 모든 대기(정지)차량의 대기시간 총합 )

    return reward
    


# TraCI 인터페이스를 사용하여 진행중인 시뮬레이션의 신호등 신호길이를 실시간으로 변경하는 함수
# tlsID : 신호등 ID,  phaseIndex : 해당 신호등의 신호 phase 중 하나(e.g. 0~7 중 하나), duration : 해당 신호 phase에 적용할 새로운 신호길이(sec)
def setSig1(tlsID, phaseIndex, duration):
    ## define function that sets traffic signal of traffic light with id 'tlsID' to given parameters ##
    
    # tlsID = traci.trafficlight.getIDList()[0]  # 첫 번째 신호등(현 케이스: 교차로에 존재하는 유일한 신호등)의 ID를 가져옴
    
    program = traci.trafficlight.getAllProgramLogics(tlsID)
    logic = program[0]    # 신호등의 현재 논리(제어 방식)를 가져옴
    
    # 'phaseIndex' 단계의 지속 시간을 duration초로 변경하고 minDur와 maxDur 설정
    logic.phases[phaseIndex].duration = duration   # 지속시간을 duration으로 변경
    logic.phases[phaseIndex].minDur = duration
    logic.phases[phaseIndex].maxDur = duration
    # print(logic.getPhases())
    
    # 변경된 프로그램을 설정합니다.
    traci.trafficlight.setProgramLogic(tlsID, logic)  # 변경된 논리를 다시 설정함
    return

# TraCI 인터페이스를 사용하여 진행중인 시뮬레이션의 신호등 현 신호단계를 실시간으로 다른 신호단계로 변경하는 함수
def setSig2(tlsID, phaseIndex):
    traci.trafficlight.setPhase(tlsID, phaseIndex)
    return


            
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
