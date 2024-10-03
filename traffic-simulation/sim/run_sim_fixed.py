from multiprocessing import Process, Queue
from time import sleep, time
from collections import defaultdict

from environment import load_and_run_simulation

def run_game():

    test_duration_seconds = 600
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    start_time = time()

    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
    errors = []

    p = Process(target=load_and_run_simulation, args=(configuration_file,
                                                        start_time,
                                                        test_duration_seconds,
                                                        random,
                                                        input_queue,
                                                        output_queue,
                                                        error_queue))
    
    p.start()

    # Wait for the simulation to start
    sleep(0.2)

    # For logging
    actions = {}
    previous_next_signals = None

    while True:

        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        # Initialize per-leg data
        legs = [leg.name for leg in state.legs]
        queue_length = defaultdict(int)
        num_vehicles = defaultdict(int)

        # Process vehicles
        for vehicle in state.vehicles:
            leg = vehicle.leg
            num_vehicles[leg] += 1
            if vehicle.speed < 0.5:  # Vehicle is considered queued
                queue_length[leg] += 1

        # Get current phase Pt
        Pt = tuple(sorted([signal.name for signal in state.signals if signal.state.lower() == 'green']))

        # Get next phase Pt+1
        if previous_next_signals is not None:
            Pt_plus_1 = tuple(sorted([signal for signal, state in previous_next_signals.items() if state.lower() == 'green']))
        else:
            Pt_plus_1 = Pt  # If we don't have previous next_signals, assume next phase is same as current

        # Print the features
        print(f"\nSimulation Tick: {state.simulation_ticks}")
        print(f"Total Score: {state.total_score}")

        print("Features:")
        for leg in legs:
            print(f"  Leg: {leg}")
            print(f"    Queue Length (q_t): {queue_length[leg]}")
            print(f"    Number of Vehicles (v_t): {num_vehicles[leg]}")
            # We can't accurately compute Total Waiting Time (w_t) without vehicle IDs or waiting times

        print(f"Current Phase (P_t): {Pt}")
        print(f"Next Phase (P_t+1): {Pt_plus_1}")

        # Insert your own logic here to parse the state and 
        # select the next action to take

        signal_logic_errors = None
        prediction = {}
        prediction["signals"] = []

        # Example logic to decide next signals
        if state.simulation_ticks % 10 < 5:
            prediction["signals"].append({"name": "A1", "state": "green"})
            prediction["signals"].append({"name": "A1LeftTurn", "state": "green"})
            prediction["signals"].append({"name": "A2", "state": "green"})
            prediction["signals"].append({"name": "A2LeftTurn", "state": "green"})
            prediction["signals"].append({"name": "B1", "state": "red"})
            prediction["signals"].append({"name": "B1LeftTurn", "state": "red"})
            prediction["signals"].append({"name": "B2", "state": "red"})
            prediction["signals"].append({"name": "B2LeftTurn", "state": "red"})
        else:
            prediction["signals"].append({"name": "A1", "state": "red"})
            prediction["signals"].append({"name": "A1LeftTurn", "state": "red"})
            prediction["signals"].append({"name": "A2", "state": "red"})
            prediction["signals"].append({"name": "A2LeftTurn", "state": "red"})
            prediction["signals"].append({"name": "B1", "state": "green"})
            prediction["signals"].append({"name": "B1LeftTurn", "state": "green"})
            prediction["signals"].append({"name": "B2", "state": "green"})
            prediction["signals"].append({"name": "B2LeftTurn", "state": "green"})

        # Update the desired phase of the traffic lights
        next_signals = {}
        current_tick = state.simulation_ticks

        for signal in prediction['signals']:
            actions[current_tick] = (signal['name'], signal['state'])
            next_signals[signal['name']] = signal['state']

        input_queue.put(next_signals)

        # Store next_signals for next iteration
        previous_next_signals = next_signals.copy()

    # End of simulation, return the score

    # Transform the score to the range [0, 1]
    if state.total_score == 0:
        state.total_score = 1e9

    inverted_score = 1. / state.total_score

    return inverted_score

if __name__ == '__main__':
    run_game()
