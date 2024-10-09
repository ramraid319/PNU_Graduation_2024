import torch
import matplotlib.pyplot as plt
import DQN
import Simulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Now the machine learning will utilize [{device}] on this PC.")

# 파라미터 설정
env_name = 'CARLA'  # 또는 'CARLA'
episodes = 5 # default : 1000
sync_interval = 20
action_size = 4  # 환경에 맞는 액션 크기 설정

# 환경 및 에이전트 설정
env = Simulator.make(env_name)
agent = DQN.Agent(action_size, device)

reward_history = []

# 그래프 설정
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Live Updating Graph of Total Reward')

# 학습 루프

# for episode in range(episodes):
#     state = env.start()
#     done = False
#     total_reward = 0

#     while not done:
#         action = agent.get_action(state)
#         next_state, reward, done = env.step(action)
#         agent.update(state, action, reward, next_state, done)
#         state = next_state
#         total_reward += reward

#     if episode % sync_interval == 0:
#         agent.sync_qnet()

#     reward_history.append(total_reward)
#     print(f"Episode {episode}, Total Reward: {total_reward}")

#     # 그래프 업데이트
#     line.set_xdata(range(0, len(reward_history)))
#     line.set_ydata(reward_history)
#     ax.relim()
#     ax.autoscale_view()
#     plt.draw()
#     plt.pause(0.1)

#     agent.decay_epsilon()

#     torch.save(agent.qnet.state_dict(), 'model.pth')

try:
    for episode in range(episodes):
        state = env.start()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()

        if episode < episodes:
            env.reset()            

        reward_history.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")


        # 그래프 업데이트
        line.set_xdata(range(0, len(reward_history)))
        line.set_ydata(reward_history)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

        agent.decay_epsilon()

        torch.save(agent.qnet.state_dict(), 'model.pth')
except:
    KeyboardInterrupt

finally:
    env.reset()


plt.ioff()
plt.show()