from multiprocessing import Process, Queue
from time import sleep, time
from environment import load_and_run_simulation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, action_size, num_signal_groups):
        super(DQN, self).__init__()
        self.num_signal_groups = num_signal_groups
        self.action_size = action_size

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size * num_signal_groups)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten to [batch_size, num_signal_groups * 5] => [1, 40]
        x = F.relu(self.fc1(x))      # [1, hidden_size]
        x = F.relu(self.fc2(x))      # [1, hidden_size]
        x = self.fc3(x)              # [1, action_size * num_signal_groups]
        x = x.view(batch_size, self.num_signal_groups, self.action_size)  # [1, num_signal_groups, action_size]
        return x

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        
    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
            
    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = idx
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = leaf - self.capacity + 1
        return (leaf, self.tree[leaf], self.data[data_idx])
    
    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.capacity = capacity
        
    def push(self, transition):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, transition)
        
    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total_priority / batch_size
        priorities = []
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
            
        sampling_probabilities = priorities / self.tree.total_priority
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight
    
    def update_priorities(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority)
            
    def __len__(self):
        return self.tree.n_entries



class Agent:
    def __init__(self, input_size, hidden_size, action_size, num_signal_groups, device):
        self.num_signal_groups = num_signal_groups
        self.action_size = action_size
        self.device = device
        
        self.policy_net = DQN(input_size, hidden_size, action_size, num_signal_groups).to(device)
        self.target_net = DQN(input_size, hidden_size, action_size, num_signal_groups).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = PrioritizedReplayMemory(10000)

        
        self.steps_done = 0
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 200
        self.gamma = 0.99
        self.batch_size = 128
        self.target_update = 10
        self.beta_start = 0.4
        self.beta_frames = 100000
        self.frame = 1

        self.save_path = "dqn_model.pth"  # Path to save the model

    def save_model(self):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            # Add other attributes you might want to save
        }, self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_model(self):
        if not os.path.exists(self.save_path):
            print(f"No saved model found at {self.save_path}")
            return

        checkpoint = torch.load(self.save_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.target_net.eval()
        print(f"Model loaded from {self.save_path}")


    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state.to(self.device)) 
                actions = q_values.max(2)[1].squeeze(0)  
                return actions.cpu().numpy()
        else:
            return np.random.randint(0, self.action_size, size=self.num_signal_groups)

        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        transitions, idxs, is_weights = self.memory.sample(self.batch_size, beta)
        
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        
        # Forward pass to get Q-values for the current states
        policy_net_output = self.policy_net(state_batch)
        action_batch_indices = action_batch.long()
        
        # Gather the Q-values corresponding to the chosen actions
        state_action_values = policy_net_output.gather(2, action_batch_indices.unsqueeze(-1)).squeeze(-1)
        
        # Compute the expected Q values using the target network
        next_state_values = torch.zeros(self.batch_size, self.num_signal_groups, device=self.device)
        if non_final_next_states.size(0) > 0:
            next_state_q_values = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = next_state_q_values.max(2)[0].detach()
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.unsqueeze(1)
        
        # Compute the loss with importance-sampling weights
        loss = (F.mse_loss(state_action_values, expected_state_action_values, reduction='none') * torch.tensor(is_weights, device=self.device)).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        # Update priorities
        priorities = (state_action_values - expected_state_action_values).abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(idxs, priorities)
        
        print(f"Training Step: {self.steps_done}, Loss: {loss.item()}")



def get_state_features(state, prev_cur_phase, time_this_phase, vehicle_waiting_time):
    NUM_SIGNAL_GROUPS = len(state.signal_groups)
    signal_group_names = state.signal_groups
    signal_name_to_idx = {name: idx for idx, name in enumerate(signal_group_names)}

    queue_length = np.zeros(NUM_SIGNAL_GROUPS)
    waiting_time = np.zeros(NUM_SIGNAL_GROUPS)
    num_of_vehicles = np.zeros(NUM_SIGNAL_GROUPS)
    cur_phase = np.zeros(NUM_SIGNAL_GROUPS)

    for signal in state.signals:
        idx = signal_name_to_idx[signal.name]
        if signal.state == 'green':
            cur_phase[idx] = 1
        else:
            cur_phase[idx] = 0

    leg_name_to_signal_groups = {}
    for leg in state.legs:
        leg_name = leg.name
        signal_groups = leg.signal_groups
        leg_name_to_signal_groups[leg_name] = [signal_name_to_idx[sg] for sg in signal_groups]
        
    for vehicle in state.vehicles:
        leg_name = vehicle.leg
        signal_group_indices = leg_name_to_signal_groups.get(leg_name, [])
        for idx in signal_group_indices:
            num_of_vehicles[idx] += 1

            if vehicle.speed < 0.1:
                queue_length[idx] += 1

            vehicle_id = vehicle.id
            vehicle_wait = vehicle_waiting_time.get(vehicle_id, 0)
            waiting_time[idx] += vehicle_wait

    for i in range(NUM_SIGNAL_GROUPS):
        if cur_phase[i] == prev_cur_phase[i]:
            time_this_phase[i] += 1
        else:
            time_this_phase[i] = 1

    state_features = {
        'waiting_time': waiting_time,
        'queue_length': queue_length,
        'num_of_vehicles': num_of_vehicles,
        'cur_phase': cur_phase,
        'time_this_phase': time_this_phase.copy()
    }

    return state_features

def compute_reward(state_features, next_state_features):
    reward = -np.sum(next_state_features['waiting_time'])
    phase_changes = np.sum(state_features['cur_phase'] != next_state_features['cur_phase'])
    reward -= 0.1 * phase_changes

    return reward

def run_game():
    NUM_SIGNAL_GROUPS = 8
    NUM_ACTIONS = 2
    STATE_FEATURES = 5  # queue_length, num_of_vehicles, cur_phase, time_this_phase, waiting_time
    input_size = NUM_SIGNAL_GROUPS * STATE_FEATURES  # 8 * 5 = 40
    hidden_size = 256  # You can choose a different value if desired
    action_size = NUM_ACTIONS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(input_size=input_size, hidden_size=hidden_size, action_size=action_size, num_signal_groups=NUM_SIGNAL_GROUPS, device=device)
    actions = {}
    
    num_episodes = 100

    for episode in range(num_episodes):
        print(f"Episode {episode}\n")
        test_duration_seconds = 600
        random_state = True
        configuration_file = "models/1/glue_configuration.yaml"
        start_time = time()
        input_queue = Queue()
        output_queue = Queue()
        error_queue = Queue()
        errors = []
        p = Process(target=load_and_run_simulation, args=(configuration_file,
                                                        start_time,
                                                        test_duration_seconds,
                                                        random_state,
                                                        input_queue,
                                                        output_queue,
                                                        error_queue))
        p.start()
        sleep(0.2)
        state = None
        prev_cur_phase = np.zeros(NUM_SIGNAL_GROUPS)
        time_this_phase = np.zeros(NUM_SIGNAL_GROUPS)
        total_reward = 0

        while True:
            state = output_queue.get()
            if state.is_terminated:
                p.join()
                break
            
            vehicle_waiting_time = state.vehicle_waiting_time
            state_features = get_state_features(state, prev_cur_phase, time_this_phase, vehicle_waiting_time)

            prev_cur_phase = state_features['cur_phase'].copy()
            
            queue_length = state_features['queue_length']
            num_of_vehicles = state_features['num_of_vehicles']
            cur_phase = state_features['cur_phase']
            time_this_phase = state_features['time_this_phase']
            waiting_time = state_features['waiting_time']
            
            state_tensor = torch.from_numpy(np.stack([
                queue_length,
                num_of_vehicles,
                cur_phase,
                time_this_phase,
                waiting_time
            ])).float().unsqueeze(0).to(device)  # Shape: [1, NUM_SIGNAL_GROUPS * STATE_FEATURES] = [1, 40]
            
            actions_array = agent.select_action(state_tensor)
            
            next_signals = {}
            for idx in range(NUM_SIGNAL_GROUPS):
                group_name = state.signal_groups[idx]
                current_state = state.signals[idx].state
                action = actions_array[idx]
                if action == 1 and state_features['time_this_phase'][idx] >= 6:
                    desired_state = 'green' if current_state != 'green' else 'red'
                else:
                    desired_state = current_state
                next_signals[group_name] = desired_state

            input_queue.put(next_signals)
            next_state = output_queue.get()

            if next_state.is_terminated:
                p.join()
                break

            next_vehicle_waiting_time = next_state.vehicle_waiting_time
            next_state_features = get_state_features(next_state, prev_cur_phase, time_this_phase, next_vehicle_waiting_time)
            reward = compute_reward(state_features, next_state_features)
            total_reward += reward
            reward_tensor = torch.tensor([reward], dtype=torch.float).to(device)

            # Construct next state tensor
            next_state_tensor = torch.from_numpy(np.stack([
                next_state_features['queue_length'],
                next_state_features['num_of_vehicles'],
                next_state_features['cur_phase'],
                next_state_features['time_this_phase'],
                next_state_features['waiting_time']
            ])).float().unsqueeze(0).to(device)  # Shape: [1, 40]

            action_tensor = torch.from_numpy(actions_array).unsqueeze(0).to(device)  # Shape: [1, NUM_SIGNAL_GROUPS]
            
            # Store transition in memory
            agent.memory.push(Transition(state_tensor, action_tensor, next_state_tensor, reward_tensor))
            agent.optimize_model()
            
            state = next_state
            state_features = next_state_features
            if agent.steps_done % agent.target_update == 0:
                agent.update_target_network()
        
        if state.is_terminated:
            print(f"Episode Finished. Total Reward: {total_reward}")

<<<<<<< HEAD
    agent.save_model()
=======
        print(f'Vehicles: {state.vehicles}')
        print(f'Signals: {state.signals}')
>>>>>>> 71e319c409e8fcfd5bcacff2bcc7cd4290d34b54

    total_reward = 0
    if state.total_score == 0:
        state.total_score = 1e9
    inverted_score = 1. / state.total_score
    return inverted_score

if __name__ == '__main__':
    run_game()