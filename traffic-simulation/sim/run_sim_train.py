# run_sim.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, defaultdict
from multiprocessing import Process, Queue
from time import sleep, time

from environment import load_and_run_simulation

# DQNAgent class definition remains the same
class DQNAgent(nn.Module):
    # [Previous class definition remains unchanged]
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Shared layers
        self.shared_fc1 = nn.Linear(state_size, 64)
        self.shared_fc2 = nn.Linear(64, 64)

        # Phase-specific layers
        self.ns_fc1 = nn.Linear(64, 32)
        self.ns_fc2 = nn.Linear(32, action_size)

        self.we_fc1 = nn.Linear(64, 32)
        self.we_fc2 = nn.Linear(32, action_size)

        self.relu = nn.ReLU()

    def forward(self, state, phase):
        x = self.relu(self.shared_fc1(state))
        x = self.relu(self.shared_fc2(x))

        if phase == 'NS':
            x = self.relu(self.ns_fc1(x))
            q_values = self.ns_fc2(x)
        else:  # phase == 'WE'
            x = self.relu(self.we_fc1(x))
            q_values = self.we_fc2(x)

        return q_values

def run_game():
    test_duration_seconds = 600
    random_simulation = True
    configuration_file = "models/1/glue_configuration.yaml"
    start_time = time()

    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
    errors = []

    p = Process(target=load_and_run_simulation, args=(configuration_file,
                                                      start_time,
                                                      test_duration_seconds,
                                                      random_simulation,
                                                      input_queue,
                                                      output_queue,
                                                      error_queue))

    p.start()

    # Wait for the simulation to start
    sleep(0.2)

    # Initialize the DQN agent
    legs = ['A1', 'A2', 'B1', 'B2']  # Define your legs
    num_legs = len(legs)
    state_size = num_legs * 2 + 4  # Queue lengths, number of vehicles, current phase (2), next phase (2)
    action_size = 2  # Keep phase or change phase

    agent = DQNAgent(state_size, action_size)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Experience replay memory
    memory = deque(maxlen=5000)
    batch_size = 64

    # Epsilon-greedy parameters
    epsilon = 1.0        # Exploration rate
    epsilon_min = 0.05   # Minimum exploration rate
    epsilon_decay = 0.995  # Decay rate for epsilon
    gamma = 0.99         # Discount factor for future rewards

    # Target network for stability
    target_agent = DQNAgent(state_size, action_size)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.eval()

    # Other variables
    previous_total_score = None
    lambda_value = 5  # Penalty for changing phase
    step_count = 0
    target_update_freq = 1000  # Update target network every 1000 steps

    # Phase durations and variables
    GREEN_PHASE_DURATION = 6  # Minimum green phase duration (as per constraints)
    current_phase_time = 0
    current_phase = 'NS'  # Start with NS phase

    # Function to select action using epsilon-greedy policy remains the same
    def select_action(state_vector, phase, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(action_size)  # Random action
        else:
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            with torch.no_grad():
                q_values = agent(state_tensor, phase)
            return np.argmax(q_values.cpu().data.numpy())

    # Helper function to get signals for phases
    def get_signals_for_phase(phase):
        if phase == 'NS':
            return {
                'A1': 'green',
                'A1LeftTurn': 'green',
                'A2': 'green',
                'A2LeftTurn': 'green',
                'B1': 'red',
                'B1LeftTurn': 'red',
                'B2': 'red',
                'B2LeftTurn': 'red'
            }
        elif phase == 'WE':
            return {
                'A1': 'red',
                'A1LeftTurn': 'red',
                'A2': 'red',
                'A2LeftTurn': 'red',
                'B1': 'green',
                'B1LeftTurn': 'green',
                'B2': 'green',
                'B2LeftTurn': 'green'
            }

    # Set initial signals
    signal_states = get_signals_for_phase(current_phase)
    prediction = {
        "signals": [{"name": k, "state": v} for k, v in signal_states.items()]
    }
    next_signals = {signal['name']: signal['state'] for signal in prediction['signals']}
    input_queue.put(next_signals)
    print(f"Initial signals set to phase: {current_phase}")

    while True:
        # Get the current state from the simulation
        sim_state = output_queue.get()

        if sim_state.is_terminated:
            p.join()
            break

        # Extract features and construct state_vector
        queue_length = defaultdict(int)
        num_vehicles = defaultdict(int)

        for vehicle in sim_state.vehicles:
            leg = vehicle.leg
            num_vehicles[leg] += 1
            if vehicle.speed < 0.5:
                queue_length[leg] += 1

        # Update timers
        current_phase_time += 1

        # Check if minimum green phase duration has been met
        if current_phase_time >= GREEN_PHASE_DURATION:
            # Agent can decide to change phase
            # Construct the state vector
            state_vector = []
            max_queue_length = 10  # Adjust based on expected max values
            max_num_vehicles = 20  # Adjust based on expected max values

            for leg in legs:
                q_len = queue_length[leg]
                num_veh = num_vehicles[leg]
                # Normalize features
                state_vector.append(q_len / max_queue_length)
                state_vector.append(num_veh / max_num_vehicles)

            # Encode current phase
            if current_phase == 'NS':
                state_vector.extend([1, 0])  # [1, 0] for NS
            else:
                state_vector.extend([0, 1])  # [0, 1] for WE

            # Encode next phase
            next_phase_temp = 'WE' if current_phase == 'NS' else 'NS'
            if next_phase_temp == 'NS':
                state_vector.extend([1, 0])
            else:
                state_vector.extend([0, 1])

            action = select_action(state_vector, current_phase, epsilon)
            phase_changes = (action == 1)
            if phase_changes:
                next_phase = 'WE' if current_phase == 'NS' else 'NS'
                current_phase = next_phase
                current_phase_time = 0
                # Set signals for the new phase
                signal_states = get_signals_for_phase(current_phase)
                prediction = {
                    "signals": [{"name": k, "state": v} for k, v in signal_states.items()]
                }
                next_signals = {signal['name']: signal['state'] for signal in prediction['signals']}
                input_queue.put(next_signals)
                print(f"Phase changed to: {current_phase}")

                # Calculate reward for phase change
                if previous_total_score is not None:
                    delta_total_score = sim_state.total_score - previous_total_score
                    reward = -delta_total_score - lambda_value
                else:
                    reward = 0  # First step
                previous_total_score = sim_state.total_score

                # Store experience
                done = sim_state.is_terminated
                next_state_vector = state_vector  # Placeholder
                memory.append((state_vector, action, reward, next_state_vector, done, current_phase))

                # Training step
                if len(memory) >= batch_size:
                    minibatch = random.sample(memory, batch_size)
                    for s, a, r, s_next, d, phase in minibatch:
                        state_tensor = torch.FloatTensor(s).unsqueeze(0)
                        next_state_tensor = torch.FloatTensor(s_next).unsqueeze(0)

                        # Compute target Q-value
                        target = r
                        if not d:
                            with torch.no_grad():
                                next_q_values = target_agent(next_state_tensor, phase)
                                target += gamma * torch.max(next_q_values).item()

                        # Get current Q-value
                        q_values = agent(state_tensor, phase)
                        current_q = q_values[0][a]

                        # Compute loss
                        target_tensor = torch.tensor([target], dtype=torch.float32)
                        loss = criterion(current_q, target_tensor)

                        # Optimize the model
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Update epsilon
                    if epsilon > epsilon_min:
                        epsilon *= epsilon_decay

                    # Update target network
                    if step_count % target_update_freq == 0:
                        target_agent.load_state_dict(agent.state_dict())

                step_count += 1

                continue  # Skip to the next iteration

        # If not changing phase, do not send any signals
        # Let the simulation continue with current signals
        # Also, we need to calculate the reward even when not changing phase
        if previous_total_score is not None:
            delta_total_score = sim_state.total_score - previous_total_score
            reward = -delta_total_score
        else:
            reward = 0  # First step
        previous_total_score = sim_state.total_score

        # Store experience
        done = sim_state.is_terminated
        next_state_vector = state_vector  # Placeholder
        action = 0  # Since we didn't change phase, action is 0
        memory.append((state_vector, action, reward, next_state_vector, done, current_phase))

        # Training step
        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)
            for s, a, r, s_next, d, phase in minibatch:
                state_tensor = torch.FloatTensor(s).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(s_next).unsqueeze(0)

                # Compute target Q-value
                target = r
                if not d:
                    with torch.no_grad():
                        next_q_values = target_agent(next_state_tensor, phase)
                        target += gamma * torch.max(next_q_values).item()

                # Get current Q-value
                q_values = agent(state_tensor, phase)
                current_q = q_values[0][a]

                # Compute loss
                target_tensor = torch.tensor([target], dtype=torch.float32)
                loss = criterion(current_q, target_tensor)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # Update target network
            if step_count % target_update_freq == 0:
                target_agent.load_state_dict(agent.state_dict())

        step_count += 1

        # Optional: Print progress
        if step_count % 100 == 0:
            print(f"Step: {step_count}, Epsilon: {epsilon:.4f}, Reward: {reward:.2f}")

    # End of simulation
    print("Simulation ended. Saving the model...")
    torch.save(agent.state_dict(), 'dqn_agent.pth')
    print("Model saved as 'dqn_agent.pth'.")

if __name__ == '__main__':
    run_game()
