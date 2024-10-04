from multiprocessing import Process, Queue
from time import sleep, time
import logging
from queue import Empty
import os
from environment import load_and_run_simulation

# Get home directory
log_file_path = os.path.expanduser('~/Downloads/DM-i-AI-2024-master/traffic-simulation/simulation_log.log')

# Configure logging to a file with an absolute path
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    filename=log_file_path,  # Absolute path to log file
    filemode='w'  # 'w' for overwrite, 'a' for append
)

print(f"Logging to file: {log_file_path}")


def set_signals_to_green(signal_names):
    return [{"name": signal, "state": "green"} for signal in signal_names]

def set_signals_to_red(signal_names):
    return [{"name": signal, "state": "red"} for signal in signal_names]

def run_game():

    # Configuration parameters
    test_duration_seconds = 300  # 10 minutes
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    simulation_start_time = time()

    # Initialize queues for inter-process communication
    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
    errors = []

    # Start the simulation process
    p = Process(target=load_and_run_simulation, args=(
        configuration_file,
        simulation_start_time,
        test_duration_seconds,
        random,
        input_queue,
        output_queue,
        error_queue
    ))
    p.start()

    # Wait briefly to ensure the simulation starts
    sleep(0.2)

    # Define signal groups
    group_a_signals = ["A1", "A2", "A1LeftTurn", "A2LeftTurn"]
    group_b_signals = ["B1", "B2", "B1LeftTurn", "B2LeftTurn"]

    # Initialize state machine variables
    current_state = 'A_wait'  # Initial state
    state_start_time = None    # To be set on receiving the first state

    while True:
        try:
            # Attempt to retrieve the latest state from the simulation
            state = output_queue.get(timeout=1)  # Adjust timeout as needed
            current_time = time()

            if state_start_time is None:
                state_start_time = current_time  # Initialize on first state received
                logging.info(f"Initial state set to '{current_state}' at time {state_start_time}")

            if state.is_terminated:
                p.join()
                logging.info("Simulation terminated.")
                break

            # Log the current state information
            logging.debug(f'Current Time: {current_time}')
            logging.debug(f'Simulation Tick: {state.simulation_ticks}')
            logging.debug(f'Vehicles: {state.vehicles}')
            logging.debug(f'Signals: {state.signals}')
            logging.debug(f'Total Score: {state.total_score}')

            # Calculate elapsed time since the current state started
            elapsed_time = current_time - state_start_time
            logging.debug(f'Elapsed Time in "{current_state}": {elapsed_time:.2f} seconds')

            signal_logic_errors = None
            prediction = {}
            prediction["signals"] = []

            # State machine logic based on elapsed time
            if current_state == 'A_wait':
                if elapsed_time >= 1:
                    # Transition to A_green
                    logging.info("Transitioning to 'A_green'. Setting Group A signals to GREEN.")
                    prediction["signals"] = set_signals_to_green(group_a_signals)
                    current_state = 'A_green'
                    state_start_time = current_time
            elif current_state == 'A_green':
                if elapsed_time >= 20:
                    # Transition to B_wait
                    logging.info("Transitioning to 'B_wait'. Setting Group A signals to RED.")
                    prediction["signals"] = set_signals_to_red(group_a_signals)
                    current_state = 'B_wait'
                    state_start_time = current_time
            elif current_state == 'B_wait':
                if elapsed_time >= 3:
                    # Transition to B_green
                    logging.info("Transitioning to 'B_green'. Setting Group B signals to GREEN.")
                    prediction["signals"] = set_signals_to_green(group_b_signals)
                    current_state = 'B_green'
                    state_start_time = current_time
            elif current_state == 'B_green':
                if elapsed_time >= 20:
                    # Transition back to A_wait
                    logging.info("Transitioning to 'A_wait'. Setting Group B signals to RED.")
                    prediction["signals"] = set_signals_to_red(group_b_signals)
                    current_state = 'A_wait'
                    state_start_time = current_time
            else:
                logging.warning(f"Unknown state: {current_state}")

            # Update the desired phase of the traffic lights
            next_signals = {}
            current_tick = state.simulation_ticks

            for signal in prediction['signals']:
                next_signals[signal['name']] = signal['state']

            signal_logic_errors = input_queue.put(next_signals)

            if signal_logic_errors:
                errors.append(signal_logic_errors)

            # Handle any errors from the simulation
            try:
                while True:
                    error = error_queue.get_nowait()
                    errors.append(error)
                    logging.error(f"Error received from simulation: {error}")
            except Empty:
                pass  # No more errors

        except Empty:
            logging.debug("No state received within the timeout period.")
            continue

    # End of simulation, return the score
    if state.total_score == 0:
        state.total_score = 1e9  # Prevent division by zero

    inverted_score = 1. / state.total_score
    logging.info(f"Final Inverted Score: {inverted_score}")
    return inverted_score

if __name__ == '__main__':
    try:
        run_game()
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Terminating simulation.")

