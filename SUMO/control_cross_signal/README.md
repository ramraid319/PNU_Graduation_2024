# 시뮬레이션 실행 방법

1. SUMO(Simulation of Urban MObility)를 설치 <https://sumo.dlr.de/docs/Downloads.php>

2. control_cross_signal.zip 다운받고 압축해제

3. 해당 폴더에서 명령프롬프트 실행 후 -> python control_traffic_lights.py 실행

4. SUMO 창이 실행되면, Delay (ms)에 적당한 수(ex. 80)을 입력

5. viewsetting 변경 방법(생략해도 시뮬은 실행됩니다)
   
   (1) 아래 무지개색 원판 버튼(Edit Coloring Schemes) 클릭

   (2) 창이 뜨면 폴더열리는 모양의 Import View Settings 버튼 클릭

   (3) 아까 받은 control_cross_signal 폴더의 viewsetting1.xml 파일 클릭 후 OK 클릭

   (4) OK 클릭

6. 녹색 삼각형 버튼 클릭해서 시뮬레이션 실행



# 시뮬레이션 실행 中 창/조작 설명

1. 기본적으로 SUMO창에서 시뮬레이션이 자동으로 진행됨

2. 시뮬레이션 실행시 SUMO Simulation Control 이라는 작은 창 하나가 더 뜸

3. 이 창에는 시뮬레이션의 매 step마다 각 차로별 차량 대기정보가 출력됨

   * 위에서부터 Lane E0_0, Lane E0_1, ..., Lane L12_2는 12개의 교차로 진입 차로를 의미
 
   * 이는 각각, 시뮬레이션상에서 교차로 북쪽 도로의 진입 3개 차로중 마지막 차로(가장 왼쪽차로)부터 시작해 시계방향으로 위치한 차로들에 해당

           Avg. Waiting Time:  차로에 정지중인 차량들의 평균 대기시간
           
           Max. Waiting Time:  차로에 정지중인 차량들 중 최장 대기시간
          
           Waiting Car Count:  차로에 정지중인 차량 수
    


4. 이 창의 맨 아래 입력칸에 명령어 입력하여 시뮬레이션의 신호등 조작 가능
   
     * setSig {phaseNo} {duration}
       
       : phaseNo번째 신호주기의 길이를 duration초로 변경
       
         ( phaseNo : 0~7, duration : sec(초) 로 입력 )
       
     * exit
       
       : 시뮬레이션 종료(이렇게 종료하거나, 혹은 시뮬레이션 자체 설정된 시간이 지나면 알아서 종료됨)


  ----


  # 파일 설명

- control_traffic_lights.py

  전체 시뮬레이션을 실행 / traCI로 제어 / 명령어 입력 쓰레드 / 대기차량 통계 출력 쓰레드 등을 구현한 코드입니다. (시뮬 실행시 해당 파일만 실행하시면 됩니다)

- cross.net.xml

  차량이 주행하는 도로망을 설계(edge, lane, tlLogic(신호등), junction, connection)한 파일입니다.

- random_routes.rou.xml (control_traffic_lights.py 실행시 자동생성)

  도로망을 주행할 차량 타입과 각 차량이 이동할 수 있는 route를 정의하고, 각 루트에 random하게 차량들이 생성 및 주행할 수 있도록 설정하는 파일

- cross.sumocfg

  sumo configuration file로 시뮬레이션에서 사용될 네트워크파일(.net.xml)과 루트파일(.rou.xml)을 지정해주고, 시뮬레이션이 진행될 총 시간(스텝)을 정해줍니다.

- viewsetting1.xml

  시뮬레이션 실행시 보기설정인 viewsetting을 설정한 파일입니다. (자세하게 현재 신호, 신호등번호, 신호phase, 교차로 커넥션 등을 볼수 있음)


----

  # API 관련
  
control_traffic_lights.py 파일에서 getEachLaneWaitingStats() 함수(차로별 차량대기 정보를 받기)와 setSig() 함수(특정 신호주기의 duration을 변경하기) 부분을 활용하면 될 것 같습니다.
     
