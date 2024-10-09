import torch
import DQN
import Simulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 환경 및 학습된 에이전트 설정
env = Simulator.make('SUMO')  # 또는 'CARLA'
agent = DQN.Agent(action_size=4, device=device)

# 학습된 모델 로드
agent.qnet.load_state_dict(torch.load('model.pth'))
agent.qnet.eval()

state = env.start()
done = False

# 신호 제어 루프
while not done:
    action = agent.get_action(state)
    next_state, reward, done = env.step(action)
    state = next_state

    # 환경의 상태를 신호 제어에 맞게 출력하거나 기록
    print(f"Action: {action}, Reward: {reward}")