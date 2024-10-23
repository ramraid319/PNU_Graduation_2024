import numpy as np
import matplotlib.pyplot as plt

def get_epsilon_exponential(episode):
    epsilon_max = 0.9
    epsilon_min = 0.1
    decay_steps = 800
    decay_rate = np.log(epsilon_min / epsilon_max) / (-decay_steps)  # 여기서 부호를 잘못 사용하면 증가할 수 있음

    epsilon = max(epsilon_min, epsilon_max * np.exp(decay_rate * episode))
    return epsilon

# 에피소드 범위 설정 (0부터 1000까지)
episodes = np.arange(1001)

# 엡실론 값 계산
epsilons = [get_epsilon_exponential(episode) for episode in episodes]

# 그래프 그리기
plt.plot(episodes, epsilons)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon 감소 그래프')
plt.grid(True)
plt.show()