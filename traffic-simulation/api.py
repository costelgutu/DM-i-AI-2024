import uvicorn
from fastapi import FastAPI
import datetime
import time
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto

HOST = "0.0.0.0"
PORT = 4321

app = FastAPI()
start_time = time.time()

# Define signal groups
group_a_signals = ["A1", "A2", "A1LeftTurn", "A2LeftTurn"]
group_b_signals = ["B1", "B2", "B1LeftTurn", "B2LeftTurn"]

# Define state machine variables
current_state = 'A_wait'
state_start_time = None

def set_signals_to_green(signal_names):
    return [{"name": signal, "state": "green"} for signal in signal_names]

def set_signals_to_red(signal_names):
    return [{"name": signal, "state": "red"} for signal in signal_names]

@app.get('/api')
def hello():
    return {
        "service": "traffic-simulation-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"

@app.post('/predict', response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(request: TrafficSimulationPredictRequestDto):
    global current_state, state_start_time

    
    # Decode request
    vehicles = request.vehicles
    total_score = request.total_score
    simulation_ticks = request.simulation_ticks
    signals = request.signals
    signal_groups = request.signal_groups
    legs = request.legs
    allowed_green_signal_combinations = request.allowed_green_signal_combinations
    is_terminated = request.is_terminated

    logger.info(f'Number of vehicles at tick {simulation_ticks}: {len(vehicles)}')

    # Initialize state_start_time if it's None
    current_time = time.time()
    if state_start_time is None:
        state_start_time = current_time
        logger.info(f"Initial state set to '{current_state}' at time {state_start_time}")

    # Calculate elapsed time in the current state
    elapsed_time = current_time - state_start_time
    logger.debug(f'Elapsed Time in "{current_state}": {elapsed_time:.2f} seconds')

    # State machine logic for transitioning traffic signals
    prediction = {}
    prediction["signals"] = []

    if current_state == 'A_wait':
        if elapsed_time >= 1:
            # Transition to A_green
            logger.info("Transitioning to 'A_green'. Setting Group A signals to GREEN.")
            prediction["signals"] = set_signals_to_green(group_a_signals)
            current_state = 'A_green'
            state_start_time = current_time
    elif current_state == 'A_green':
        if elapsed_time >= 15:
            # Transition to B_wait
            logger.info("Transitioning to 'B_wait'. Setting Group A signals to RED.")
            prediction["signals"] = set_signals_to_red(group_a_signals)
            current_state = 'B_wait'
            state_start_time = current_time
    elif current_state == 'B_wait':
        if elapsed_time >= 3:
            # Transition to B_green
            logger.info("Transitioning to 'B_green'. Setting Group B signals to GREEN.")
            prediction["signals"] = set_signals_to_green(group_b_signals)
            current_state = 'B_green'
            state_start_time = current_time
    elif current_state == 'B_green':
        if elapsed_time >= 15:
            # Transition back to A_wait
            logger.info("Transitioning to 'A_wait'. Setting Group B signals to RED.")
            prediction["signals"] = set_signals_to_red(group_b_signals)
            current_state = 'A_wait'
            state_start_time = current_time
    else:
        logger.warning(f"Unknown state: {current_state}")

    # Select the signals based on the state machine and respond
    response_signals = [SignalDto(name=sig["name"], state=sig["state"]) for sig in prediction["signals"]]

    # Return the signal state as the prediction
    response = TrafficSimulationPredictResponseDto(signals=response_signals)
    return response

if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
