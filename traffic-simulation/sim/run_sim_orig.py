from multiprocessing import Process, Queue
from time import sleep, time
from tabulate import tabulate  # Ensure this is installed via pip
import logging
from logging.handlers import RotatingFileHandler
# from pythonjsonlogger import jsonlogger  # Uncomment if using JSON structured logging
from environment import load_and_run_simulation

# Predefined Groups
PREDEFINED_GROUPS = {
    "Group A": ['A1', 'A1LeftTurn', 'A2', 'A2LeftTurn'],
    "Group B": ['B1', 'B1LeftTurn', 'B2', 'B2LeftTurn']
}

# Configure Logging
def setup_logging():
    """
    Sets up logging to output to both console and a rotating log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the root logger level to DEBUG to capture all levels

    # Formatter for log messages
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Uncomment the following lines to use JSON formatted logs
    # json_formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set console to display INFO and above
    console_handler.setFormatter(formatter)
    # To use JSON formatting in console
    # console_handler.setFormatter(json_formatter)
    
    # Rotating File Handler
    file_handler = RotatingFileHandler(
        'simulation.log',
        maxBytes=5*1024*1024,  # 5 MB per log file
        backupCount=3  # Keep up to 3 backup log files
    )
    file_handler.setLevel(logging.DEBUG)  # Log everything to the file
    file_handler.setFormatter(formatter)
    # To use JSON formatting in file
    # file_handler.setFormatter(json_formatter)
    
    # Add Handlers to the Logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # If using JSON structured logging, comment out the above and uncomment below
    # json_handler = RotatingFileHandler('simulation.json', maxBytes=5*1024*1024, backupCount=3)
    # json_handler.setLevel(logging.DEBUG)
    # json_handler.setFormatter(json_formatter)
    # logger.addHandler(json_handler)

# Initialize Logging
setup_logging()

def format_vehicles(vehicles):
    """
    Formats the list of vehicles into a table.
    """
    headers = ["ID", "Speed", "Distance to Stop", "Leg"]
    table = [[v.id, v.speed, v.distance_to_stop, v.leg] for v in vehicles]
    return tabulate(table, headers, tablefmt="pretty")

def format_signals(signals):
    """
    Formats the list of signals into a table.
    """
    headers = ["Name", "State"]
    table = [[s.name, s.state] for s in signals]
    return tabulate(table, headers, tablefmt="pretty")

def format_signal_groups(signal_groups):
    """
    Formats the list of signal groups into a table by dividing them into predefined groups.
    
    Args:
        signal_groups (list): A list of signal names (strings).
        
    Returns:
        str: A formatted table string representing the signal groups.
    """
    headers = ["Group Name", "Signals"]
    table = []
    
    # Assign signals to groups
    for group_name, signals in PREDEFINED_GROUPS.items():
        assigned_signals = [signal for signal in signal_groups if signal in signals]
        if assigned_signals:
            table.append([group_name, ', '.join(assigned_signals)])
        else:
            table.append([group_name, "None"])
    
    # Handle any signals not assigned to predefined groups
    ungrouped_signals = [signal for signal in signal_groups 
                         if signal not in PREDEFINED_GROUPS["Group A"] 
                         and signal not in PREDEFINED_GROUPS["Group B"]]
    
    if ungrouped_signals:
        table.append(["Ungrouped", ', '.join(ungrouped_signals)])
    
    return tabulate(table, headers, tablefmt="pretty")

class TrafficController:
    """
    A finite state machine to control traffic signals based on time.
    """
    def __init__(self, predefined_groups):
        self.predefined_groups = predefined_groups
        self.phase = 'A_wait'  # Initial phase
        self.timer = 0  # Timer for the current phase

    def update(self):
        """
        Update the state based on the current phase and timer.
        Returns a list of signal actions to perform.
        """
        self.timer += 1
        actions = []

        if self.phase == 'A_wait':
            if self.timer == 1:
                # Turn Group A to green
                logging.info("Phase Transition: Group A set to GREEN")
                for signal in self.predefined_groups["Group A"]:
                    actions.append({"name": signal, "state": "green"})
                self.phase = 'A_green'
                self.timer = 0

        elif self.phase == 'A_green':
            if self.timer == 10:
                # Turn Group A to red
                logging.info("Phase Transition: Group A set to RED")
                for signal in self.predefined_groups["Group A"]:
                    actions.append({"name": signal, "state": "red"})
                self.phase = 'B_wait'
                self.timer = 0

        elif self.phase == 'B_wait':
            if self.timer == 2:
                # Turn Group B to green
                logging.info("Phase Transition: Group B set to GREEN")
                for signal in self.predefined_groups["Group B"]:
                    actions.append({"name": signal, "state": "green"})
                self.phase = 'B_green'
                self.timer = 0

        elif self.phase == 'B_green':
            if self.timer == 10:
                # Turn Group B to red
                logging.info("Phase Transition: Group B set to RED")
                for signal in self.predefined_groups["Group B"]:
                    actions.append({"name": signal, "state": "red"})
                self.phase = 'B_end'
                self.timer = 0

        elif self.phase == 'B_end':
            if self.timer == 12:
                # Restart Group A cycle
                logging.info("Phase Transition: Restarting Group A cycle")
                self.phase = 'A_wait'
                self.timer = 0

        return actions

def run_game():
    """
    Runs the traffic simulation game with enhanced logging and a time-based traffic controller.
    """
    logging.info("Starting the traffic simulation game.")

    # Simulation Parameters
    test_duration_seconds = 600  # 10 minutes
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    start_time = time()

    # Initialize Queues for IPC
    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
    errors = []

    # Initialize Traffic Controller
    traffic_controller = TrafficController(PREDEFINED_GROUPS)

    # Start the simulation process
    logging.debug("Initializing the simulation process.")
    p = Process(target=load_and_run_simulation, args=(
        configuration_file,
        start_time,
        test_duration_seconds,
        random,
        input_queue,
        output_queue,
        error_queue
    ))
    
    p.start()
    logging.info(f"Simulation process started with PID: {p.pid}")

    # Wait briefly to allow the simulation to initialize
    sleep(0.2)
    logging.debug("Waited 0.2 seconds for simulation initialization.")

    # Dictionary to keep track of actions per tick
    actions = {}

    while True:
        try:
            # Attempt to retrieve the next state with a timeout to prevent blocking indefinitely
            state = output_queue.get(timeout=5)
            logging.debug(f"Retrieved state for tick {state.simulation_ticks}.")
        except Exception as e:
            logging.error(f"Error retrieving state from output_queue: {e}")
            p.terminate()
            logging.info("Terminated the simulation process due to error.")
            break

        if state.is_terminated:
            logging.info("Simulation has signaled termination.")
            p.join()
            logging.info("Simulation process joined successfully.")
            break

        # Log the current tick
        logging.info(f"\n--- Tick {state.simulation_ticks} ---")

        # Format and log Vehicles
        vehicles_formatted = format_vehicles(state.vehicles)
        logging.info("Vehicles:\n%s", vehicles_formatted)

        # Format and log Signals
        signals_formatted = format_signals(state.signals)
        logging.info("Signals:\n%s", signals_formatted)

        # Format and log Signal Groups
        if hasattr(state, 'signal_groups') and state.signal_groups:
            logging.debug(f"Processing signal_groups: {state.signal_groups}")
            signal_groups_formatted = format_signal_groups(state.signal_groups)
            logging.info("Signal Groups:\n%s", signal_groups_formatted)
        else:
            logging.info("Signal Groups: None")

        # Log Total Score
        logging.info("Total Score: %.2f\n", state.total_score)

        # Update Traffic Controller and get actions
        controller_actions = traffic_controller.update()

        # Prepare next_signals based on controller_actions
        next_signals = {}
        current_tick = state.simulation_ticks

        for action in controller_actions:
            signal_name = action['name']
            signal_state = action['state']
            actions[current_tick] = (signal_name, signal_state)
            next_signals[signal_name] = signal_state
            logging.debug(f"Tick {current_tick}: Action - Set {signal_name} to {signal_state}.")

        if next_signals:
            try:
                # Send the next signals to the simulation
                input_queue.put(next_signals)
                logging.debug(f"Tick {current_tick}: Sent next_signals to input_queue.")
            except Exception as e:
                logging.error(f"Error putting next_signals into input_queue: {e}")
                errors.append(e)

        # Optionally, handle errors from the error_queue
        try:
            error = error_queue.get_nowait()
            if error:
                logging.error(f"Simulation Error: {error}")
                errors.append(error)
        except:
            # No error to retrieve
            pass

    # End of simulation, process the final score
    logging.info("Simulation has ended. Processing final score.")

    # Transform the score to the range [0, 1]
    if state.total_score == 0:
        logging.warning("Total score is zero. Adjusting to prevent division by zero.")
        state.total_score = 1e9  # Prevent division by zero

    inverted_score = 1.0 / state.total_score
    logging.info("Final Inverted Score: %.9f", inverted_score)

    # Optionally, handle or log any accumulated errors
    if errors:
        logging.warning("Simulation completed with errors:")
        for err in errors:
            logging.warning(err)

    logging.info("Traffic simulation game has completed successfully.")
    return inverted_score

# Optional: Test Function for format_signal_groups
def test_format_signal_groups():
    """
    Tests the format_signal_groups function with different input structures.
    """
    test_cases = [
        # Case 1: List of strings
        ["A1", "A1LeftTurn", "A2", "A2LeftTurn", "B1", "B1LeftTurn", "B2", "B2LeftTurn"],
        
        # Case 2: List with some signals missing from predefined groups
        ["A1", "A1LeftTurn", "A2", "B1", "B2LeftTurn", "C1"],  # C1 is ungrouped
        
        # Case 3: Empty list
        [],
        
        # Case 4: All signals belong to predefined groups
        ["A1", "A1LeftTurn", "A2", "A2LeftTurn", "B1", "B1LeftTurn", "B2", "B2LeftTurn"],
        
        # Case 5: All signals are ungrouped
        ["C1", "C2", "C3"]
    ]

    for i, signal_groups in enumerate(test_cases, 1):
        logging.info(f"\n--- Test Case {i} ---")
        formatted = format_signal_groups(signal_groups)
        logging.info("Signal Groups:\n%s", formatted)

if __name__ == '__main__':
    # Uncomment the following line to run tests
    # test_format_signal_groups()
    
    run_game()
