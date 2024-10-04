from multiprocessing import Process, Queue
from time import sleep, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from environment import load_and_run_simulation

# Define the DQN architecture
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
        x = x.view(batch_size, -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(batch_size, self.num_signal_groups, self.action_size)
        return x

# Define the state feature extraction function
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

# Modify the run_game function to use the trained model
def run_game():
    NUM_SIGNAL_GROUPS = 8
    NUM_ACTIONS = 2
    STATE_FEATURES = 5  # queue_length, num_of_vehicles, cur_phase, time_this_phase, waiting_time
    input_size = NUM_SIGNAL_GROUPS * STATE_FEATURES  # 8 * 5 = 40
    hidden_size = 256  # Must match the training hidden size
    action_size = NUM_ACTIONS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = DQN(input_size=input_size, hidden_size=hidden_size, action_size=action_size, num_signal_groups=NUM_SIGNAL_GROUPS).to(device)
    
    # Load the trained model
    model_path = "dqn_model.pth"  # Ensure the path is correct
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist.")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['policy_net_state_dict'])
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # Initialize simulation parameters
    test_duration_seconds = 600
    random_state = False  # Use trained model, not random actions
    configuration_file = "models/1/glue_configuration.yaml"
    start_time = time()
    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
    errors = []
    
    # Start the simulation process
    p = Process(target=load_and_run_simulation, args=(configuration_file,
                                                      start_time,
                                                      test_duration_seconds,
                                                      random_state,
                                                      input_queue,
                                                      output_queue,
                                                      error_queue))
    p.start()
    
    # Wait for the simulation to initialize
    sleep(0.2)
    
    state = None
    prev_cur_phase = np.zeros(NUM_SIGNAL_GROUPS)
    time_this_phase = np.zeros(NUM_SIGNAL_GROUPS)
    vehicle_waiting_time = {}
    
    while True:
        try:
            state = output_queue.get(timeout=2)  # Adjust timeout as needed
        except:
            print("Timeout waiting for state. Exiting.")
            p.terminate()
            break
        
        if state.is_terminated:
            p.join()
            break
        print(f'Vehicles: {state.vehicles}')
        print(f'Signals: {state.signals}')
        print(f'Total score: {state.total_score}')
        # Extract state features
        state_features = get_state_features(state, prev_cur_phase, time_this_phase, vehicle_waiting_time)
        prev_cur_phase = state_features['cur_phase'].copy()
        
        queue_length = state_features['queue_length']
        num_of_vehicles = state_features['num_of_vehicles']
        cur_phase = state_features['cur_phase']
        time_this_phase = state_features['time_this_phase']
        waiting_time = state_features['waiting_time']
        
        # Prepare state tensor
        state_tensor = torch.from_numpy(np.stack([
            queue_length,
            num_of_vehicles,
            cur_phase,
            time_this_phase,
            waiting_time
        ])).float().unsqueeze(0).to(device)  # Shape: [1, 40]
        
        with torch.no_grad():
            q_values = model(state_tensor)  # Shape: [1, NUM_SIGNAL_GROUPS, NUM_ACTIONS]
            actions_tensor = q_values.argmax(dim=2).squeeze(0)  # Shape: [NUM_SIGNAL_GROUPS]
            actions_array = actions_tensor.cpu().numpy()
        
        # Prepare next_signals based on actions
        next_signals = {}
        for idx in range(NUM_SIGNAL_GROUPS):
            group_name = state.signal_groups[idx]
            current_state = state.signals[idx].state
            action = actions_array[idx]
            # Define your action logic here. For example:
            # action = 0: Keep current state
            # action = 1: Toggle state if time_this_phase >= threshold
            if action == 1 and state_features['time_this_phase'][idx] >= 6:
                desired_state = 'green' if current_state != 'green' else 'red'
            else:
                desired_state = current_state
            next_signals[group_name] = desired_state
        
        # Send actions to the simulation
        input_queue.put(next_signals)
    
    # End of simulation, return the score
    if state.total_score == 0:
        state.total_score = 1e9
    inverted_score = 1. / state.total_score
    print(f"Simulation completed. Inverted Score: {inverted_score}")
    return inverted_score

if __name__ == '__main__':
    run_game()
