# 강화학습 및 SUMO 시뮬레이션 실행 방법

1. SUMO(Simulation of Urban MObility)를 설치 <https://sumo.dlr.de/docs/Downloads.php> (이미 설치 완료시 생략)

2. SUMO_with_Reinforced_Learning.zip 다운받고 압축해제

3. 해당 폴더에서 명령프롬프트 실행 후 -> python model.py 실행  (또는 코드에디터(ide) 터미널에서 실행)

4. 실행시 강화학습 모델의 학습이 시작. 동시에 SUMO 창이 켜짐(자동으로 시뮬레이션이 시작).

   ** 만약 학습시 SUMO 시뮬레이터를 CLI버전으로 실행하려면

       model.py의 하단 env = sumo.make('cross.sumocfg', 'sumo') 부분의 두번째 파라미터를 'sumo' -> 'sumo-gui'로 설정

6. 잠시 뒤, 학습 중 매 episode마다 계산되는 total reward 값을 그래프로 표시하는 'Figure 1'이라는 matplotlib 그래프 창이 켜짐
  
7. 기본적으로 SUMO 창에서 Delay (ms)는 0(최대속도)으로 설정 돼 있음, 시뮬레이터를 천천히 보려면 200 같은 큰 수 입력

8. viewsetting 변경 (시뮬레이션 실행과는 직접 관계 없는 '보기'설정임, 한번 아래 설정해두면 다음번에는 안해도 됨)
   
   (1) 아래 무지개색 원판 버튼(Edit Coloring Schemes) 클릭

   (2) 창이 뜨면 폴더열리는 모양의 Import View Settings 버튼 클릭

   (3) 아까 받은 SUMO_with_Reinforced_Learning 폴더의 viewsetting1.xml 파일 클릭 후 OK 클릭

   (4) OK 클릭



  ----


  # 파일 설명

- model.py

  DQN 강화학습을 실행하는 코드입니다.

  강화학습에 사용되는 ReplayBuffer, QNet, DQNAgent 클래스들이 정의되어 있습니다.

  메인문에서 시뮬레이션 환경 및 DQNAgent 선언, 에피소드 수를 지정하고, 반복문에서 각 에피소드를 진행하면서 학습을 수행합니다. matplotlib을 사용하여 total reward를 별도 창에 출력하는 코드도 포함되어 있습니다. 


- sumo.py

  model.py의 강화학습에 필요한 SUMO시뮬레이션 환경을 정의하는 SumoEnv 클래스가 정의되어 있습니다.

  start(), reset(): SUMO 시뮬레이터 환경설정 및 시뮬레이션 시작(리셋)

  step(): 강화학습에서 받아온 action을 실행중인 sumo환경에 반영하여 실시간으로 신호체계를 변경하고, 시뮬레이션의 스텝(강화학습의 step과는 다름)을 진행하고, state와 reward를 계산하여 다시 강화학습 모델로 리턴합니다.

  get_state(): 실행중인 sumo 시뮬레이션에서 state들을 가져옵니다.

  get_reward(): 실행중인 sumo 시뮬레이션에서 특정 수치들을 이용하여 리워드를 계산하여 가져옵니다.
  

- control_traffic_lights.py

  sumo.py에서 사용하는 몇몇 함수들이 정의되어 있습니다.
  
  generateRandomRoutes2(): 시뮬레이터를 실행시키기 위해 필요한 .rou.xml 파일을 생성하는 함수

  getEachLaneWaitingStats(): traci 인터페이스를 통해, 시뮬레이션 교차로의 신호체계 차량수 속도 대기시간 같은 정보들을 가져와서 배열형태로 리턴하는 함수(현 시뮬 상황을 나타내는 state 반환)

  setSig(): traci 인터페이스를 통해 시뮬레이션 특정 신호단계의 duration을 변경하는 함수

  calculateReward(): traci 인터페이스를 통해 가져온 값들을 이용해 reward를 계산하여 넘겨주는 함수
  

- cross.net.xml

  차량이 주행하는 도로망을 설계(edge, lane, tlLogic(신호등), junction, connection)한 파일입니다.
  

- random_routes.rou.xml (이 폴더에 미포함, model.py 실행시 자동 생성)

  도로망을 주행할 차량 타입과 각 차량이 이동할 수 있는 route를 정의하고, 각 루트에 random하게 차량들이 생성 및 주행할 수 있도록 설정하는 파일


- cross.sumocfg

  sumo configuration file로 시뮬레이션에서 사용될 네트워크파일(.net.xml)과 루트파일(.rou.xml)을 지정해주고, 시뮬레이션이 진행될 총 시간(스텝)을 정해줍니다.


- viewsetting1.xml

  시뮬레이션 실행시 보기설정인 viewsetting을 설정한 파일입니다. (뷰세팅을 이 파일로 설정시 자세하게 현재 신호, 신호등번호, 신호phase, 교차로 커넥션 등을 볼수 있음)
     
