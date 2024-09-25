import torch

class SUMO:
    def __init__(self):
        self.config

    def start(self):
        # 시뮬레이터 환경설정 및 가동


        
        state = self.get_state()
        return state
    
    def step(self, action):
        # 입력받은 action 값에 따른 신호제어


        
        next_state = self.get_state()
        reward = self.get_reward()
        done = self.is_done()
        
        return next_state, reward, done
    
    def get_state(self):
        for i in range(4):
            pass
            # 24fps 기준 12프레임 진행


            # 십자 교차로 기준 4가지 뷰에 대한 화면 이미지 로드


            # 4개의 이미지를 4채널 tensor로 변환

        
        state = torch.tensor() # 16채널 tensor (channels, height, width)
        return state
    
    def get_reward(self):
        a = 1.001
        # 각 레인의 차량 대기시간 t
        # 지정한 구역에서 인식이 시작된 프레임부터 사라지는 프레임까지의 프레임 수의 차이
        t1 = 1
        t2 = 1
        t3 = 1
        t4 = 1
        
        reward = -((a ** t1) + (a ** t2) + (a ** t3) + (a ** t4))
        return reward
    
    def is_done(self):
        # 시나리오 종료 체크
        
        return True
