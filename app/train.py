import glob
import os
import traceback
import numpy as np
import torch
import matplotlib.pyplot as plt
import DQN
import Simulator
import time
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Now the machine learning will utilize [{device}] on this PC.")

# 파라미터 설정
env_name = 'CARLA'  # 'SUMO' 또는 'CARLA'
total_episodes = 2000 # default : 1000
sync_interval = 100   ###   10  ?????
action_size = 4  # 환경에 맞는 액션 크기 설정
save_interval = 10  # Save model to file every N episodes
model_save_path = f"results\{env_name.lower()}\\training\dqn_model.pth"
old_episodes_count = 0
start_episode = 0
# 환경 및 에이전트 설정
env = Simulator.make(env_name)
agent = DQN.Agent(action_size, device, total_episodes)

reward_history = []

# 그래프 설정
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)  # Large size and high DPI
line, = ax.plot([], [], color='blue', linewidth=2)  # label='Total Rewards'
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.title('DQN Total Reward by Episodes - Training', fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend(fontsize=14)



# # 모델 불러오기 (존재하면)
# if os.path.exists(model_save_path):
#     print("Loading existing model...")
#     start_episode = agent.load_model(model_save_path) + 1
# else:
#     print("No saved model found. Starting training from scratch.")
#     start_episode = 0



# # Pattern for the checkpoint files
# checkpoint_pattern = "trained_model_after_*_episodes.pth"

# # Search for the latest checkpoint file
# checkpoint_files = glob.glob(checkpoint_pattern)

# if checkpoint_files:
#     # # Sort by episode number (extract number from filename and sort in reverse order to get the latest episode)
#     checkpoint_files.sort(key=lambda f: int(f.split('_')[-2]), reverse=True)
#     # Use the latest checkpoint file
#     save_path = checkpoint_files[0]
    
#     print(f"\n-- Loading saved model from {save_path}... --")
#     old_episodes_count = agent.load(save_path)
#     time.sleep(1)
#     # print(f"-- Resuming training from episode {start_episode}. --")
# else:
#     print("\n-- No trained models found, starting from the very beginning. --")
#     old_episodes_count = 0

# Pattern for the checkpoint files
# save_path = "model.pth"

# # Search for the latest checkpoint file
# files = glob.glob(save_path)

# if files:
#     torch.load(save_path)
    
#     print(f"\n-- Loading saved model from {save_path}... --")
# else:
#     print("\n-- No trained models found, starting from the very beginning. --")

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
    plt.savefig(f'results\{env_name.lower()}\\training\dqn_total_rewards_training.png', bbox_inches='tight', dpi=300)  # Save with tight bounding box


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


episode_count = 0
try:
    for current_episode in range(start_episode, total_episodes):
        print(f"\n\n### Starting Episode {current_episode}, Epsilon: {agent.epsilon} ###")
        state = env.start()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            # print(next_state.shape)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        if current_episode % sync_interval == 0:
            agent.sync_qnet()

        env.reset()
        torch.cuda.empty_cache()

        reward_history.append(total_reward)
        print(f"### Ending Episode {current_episode}, Total Reward: {total_reward} ###")

        # time.sleep(10)

        # if episode % save_interval == 0:
        #     # 이전 체크포인트 파일 삭제
        #     last_checkpoint_file = save_path

        #     if last_checkpoint_file and os.path.exists(last_checkpoint_file):
        #         os.remove(last_checkpoint_file)
        #         print(f"Deleted old checkpoint: {last_checkpoint_file}")

        #     new_episodes_count = old_episodes_count + episode + 1
        #     agent.save(f"trained_model_after_{new_episodes_count}_episodes.pth", new_episodes_count) 
        #     print(f"Saving trained model to 'trained_model_after_{new_episodes_count}_episodes.pth'")
        #     # print(f"Saving trained model to 'checkpoint_episode{episode}.pth'")

        # last_checkpoint_file = "trained_model_after_{new_episodes_count}_episodes.pth"

        # 그래프 업데이트
        line.set_xdata(range(0, len(reward_history)))
        line.set_ydata(reward_history)
        ax.relim()
        ax.autoscale_view()

        # Set x-axis ticks to automatic
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Automatic ticks for integers

        plt.draw()
        plt.pause(0.1)

        agent.decay_epsilon()

        if (episode_count+1) % save_interval == 0:
            # 모델 저장
            agent.save_model(model_save_path, current_episode)
            print(f"Model saved after episode {current_episode}")

                # torch.save(agent.qnet.state_dict(), 'model.pth')

        episode_count += 1

except KeyboardInterrupt:
    print("Training interrupted by Keyboard.")
except Exception as e:
    print(f"Error during episode {current_episode}: {e}")
    print("Detailed traceback:")
    traceback.print_exc()  # This will print the full traceback of the error
finally:
    env.reset()

if len(reward_history) > 0:
    save_graph(reward_history)
    write_to_file(f'results\{env_name.lower()}\\training\\reward_history_training.txt', reward_history)

plt.ioff()
print("[ To End the program, please close <Figure 1> window. ]")
plt.show(block=True)
