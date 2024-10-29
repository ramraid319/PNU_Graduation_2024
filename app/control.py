import torch
import DQN.AgentInference as DQN
import Simulator
import traceback
import numpy as np
import torch
import matplotlib.pyplot as plt
import Simulator
from datetime import datetime

env_name = 'CARLA'  # 'SUMO' 또는 'CARLA'
test_fixed_signals = False  # 'True': to run fixed traffic signals / 'False': to run traffic signal inference by trained DQN model
total_episodes = 2000 # default : 100
action_size = 4  # 환경에 맞는 액션 크기 설정
start_episode = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 환경 및 학습된 에이전트 설정
env = Simulator.make(env_name)  # 또는 'CARLA'
agent = DQN.Agent(action_size, device=device)

reward_history = []

# Load the trained model for inference
agent.load_model(f"results\{env_name.lower()}\\training\dqn_model.pth")

# 그래프 설정
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)  # Large size and high DPI
line, = ax.plot([], [], color='blue', linewidth=2)  # label='Total Rewards'
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.title('DQN Total Reward by Episodes - Inference', fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend(fontsize=14)


def save_graph(reward_history):
    # After training, process and save the final graph
    start_point = (0, reward_history[0])
    end_point = (len(reward_history)-1, reward_history[-1])
    highest_point = (np.argmax(reward_history), np.max(reward_history))
    lowest_point = (np.argmin(reward_history), np.min(reward_history))

    # Mark important points
    important_points = [start_point, end_point, highest_point, lowest_point]
    labels = ['start', 'end', 'high', 'low']
    colors = ['red', 'green', 'orange', 'purple']

    for (x, y), label, color in zip(important_points, labels, colors):
        plt.scatter(x, y, color=color, zorder=5)  # Mark the point
        plt.text(x, y, f'({x}, {y:.2f}) {label}', fontsize=10, ha='left', color=color)  # Annotate the point

    # Set x-axis to be integers
    # ax.set_xticks(range(0, len(reward_history), 100))  # Adjust step as needed
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure x-axis values are integers

    # Save the figure as a high-resolution image
    plt.savefig(f'results\{env_name.lower()}\inference\dqn_total_rewards_inference.png', bbox_inches='tight', dpi=300)  # Save with tight bounding box


def write_to_file(filename, data_list):
    # Get the current date and time
    current_time = datetime.now()
    
    # Format the date and time as 'YYYY:MM:DD:hh:mm:ss.ss'
    formatted_time = current_time.strftime('%Y.%m.%d. %H:%M:%S.%f')[:-3]  # Trim to milliseconds
    
    # Create the line to write
    line = f"{formatted_time}\t\t" + ",".join(map(str, data_list)) + "\n"
    
    # Open the file in append mode and write the line
    with open(filename, 'a') as file:
        file.write(line)


try:
    for current_episode in range(start_episode, total_episodes):
        print(f"\n\n### Starting Episode {current_episode} ###")
        state = env.start()
        done = False
        total_reward = 0
        
        i = 0
        ## 48 24 48 24  / 12 = 4 2 4 2
        fixed_signals_pattern = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3]

        while not done:
            if test_fixed_signals == True:
                action = fixed_signals_pattern[i]
                i = (i+1) % len(fixed_signals_pattern)
            else:
                action = agent.get_action(state)

            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            
            # 환경의 상태를 신호 제어에 맞게 출력하거나 기록
            # print(f"Action: {action}, Reward: {reward}")
        env.reset()
        torch.cuda.empty_cache()

        reward_history.append(total_reward)
        print(f"### Ending Episode {current_episode}, Total Reward: {total_reward} ###")

        # 그래프 업데이트
        line.set_xdata(range(0, len(reward_history)))
        line.set_ydata(reward_history)
        ax.relim()
        ax.autoscale_view()

        # Set x-axis ticks to automatic
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Automatic ticks for integers

        plt.draw()
        plt.pause(0.1)


except KeyboardInterrupt:
    print("Inference interrupted by Keyboard.")
except Exception as e:
    print(f"Error during episode {current_episode}: {e}")
    print("Detailed traceback:")
    traceback.print_exc()  # This will print the full traceback of the error
finally:
    env.reset()

if len(reward_history) > 0:
    save_graph(reward_history)
    write_to_file(f'results\{env_name.lower()}\inference\\reward_history_inference.txt', reward_history)

plt.ioff()
print("[ To End the program, please close <Figure 1> window. ]")
plt.show(block=True)
