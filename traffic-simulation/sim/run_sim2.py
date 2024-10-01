from multiprocessing import Process, Queue
from time import sleep, time

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

    initialized = False
    current_phase = 'A'  # Start with phase 'A'
    phase_changing = False
    min_green_time = 6  # seconds
    amber_time = 4
    redamber_time = 2

    while True:

        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        if not initialized:
            initialized = True
            # Initialize signal timers
            signal_timers = {}
            for signal_group in state.signal_groups:
                signal_timers[signal_group] = {'state': 'red', 'time_in_state': 0}

            # Initialize signals_in_group
            signals_in_group = {'A': [], 'B': []}
            for signal_group in state.signal_groups:
                if signal_group.startswith('A'):
                    signals_in_group['A'].append(signal_group)
                elif signal_group.startswith('B'):
                    signals_in_group['B'].append(signal_group)

            # Initialize current_phase signals to green
            for signal_group in signals_in_group[current_phase]:
                signal_timers[signal_group]['state'] = 'green'
                signal_timers[signal_group]['time_in_state'] = 1
            # Initialize other signals to red
            other_phase = 'B' if current_phase == 'A' else 'A'
            for signal_group in signals_in_group[other_phase]:
                signal_timers[signal_group]['state'] = 'red'
                signal_timers[signal_group]['time_in_state'] = 1

        # Update time_in_state for all signal groups
        for signal_group, signal_info in signal_timers.items():
            signal_info['time_in_state'] += 1

        # Count vehicles on each leg
        leg_vehicle_counts = {}
        for vehicle in state.vehicles:
            leg = vehicle.leg
            if leg not in leg_vehicle_counts:
                leg_vehicle_counts[leg] = 0
            leg_vehicle_counts[leg] += 1

        # Sum up vehicles for 'A' and 'B' legs
        group_vehicle_counts = {'A': 0, 'B': 0}
        for leg, count in leg_vehicle_counts.items():
            if leg.startswith('A'):
                group_vehicle_counts['A'] += count
            elif leg.startswith('B'):
                group_vehicle_counts['B'] += count

        # Decide next_phase based on vehicle counts
        if group_vehicle_counts['A'] >= group_vehicle_counts['B']:
            next_phase = 'A'
        else:
            next_phase = 'B'

        # Check if we need to change phases
        if current_phase != next_phase and not phase_changing:
            # Check if signals have been green for at least min_green_time
            all_signals_in_current_green_long_enough = True
            for signal_group in signals_in_group[current_phase]:
                signal_info = signal_timers[signal_group]
                if signal_info['state'] == 'green':
                    if signal_info['time_in_state'] >= min_green_time:
                        pass
                    else:
                        all_signals_in_current_green_long_enough = False
                        break
            if all_signals_in_current_green_long_enough:
                phase_changing = True

        # Update the signals
        if phase_changing:
            # Handle the transition
            current_phase_signals = signals_in_group[current_phase]
            next_phase_signals = signals_in_group[next_phase]
            all_current_phase_signals_in_red = True

            # Update current phase signals
            for signal_group in current_phase_signals:
                signal_info = signal_timers[signal_group]
                state_s = signal_info['state']
                time_in_state = signal_info['time_in_state']

                if state_s == 'green' and time_in_state >= min_green_time:
                    # Transition to amber
                    state_s = 'amber'
                    time_in_state = 1
                elif state_s == 'amber' and time_in_state >= amber_time:
                    # Transition to red
                    state_s = 'red'
                    time_in_state = 1
                elif state_s == 'red':
                    pass
                signal_info['state'] = state_s
                signal_info['time_in_state'] = time_in_state
                if state_s != 'red':
                    all_current_phase_signals_in_red = False

            # Update next phase signals
            if all_current_phase_signals_in_red:
                all_next_phase_signals_in_green = True
                for signal_group in next_phase_signals:
                    signal_info = signal_timers[signal_group]
                    state_s = signal_info['state']
                    time_in_state = signal_info['time_in_state']

                    if state_s == 'red':
                        # Transition to redamber
                        state_s = 'redamber'
                        time_in_state = 1
                    elif state_s == 'redamber' and time_in_state >= redamber_time:
                        # Transition to green
                        state_s = 'green'
                        time_in_state = 1
                    elif state_s == 'green':
                        pass
                    signal_info['state'] = state_s
                    signal_info['time_in_state'] = time_in_state
                    if state_s != 'green':
                        all_next_phase_signals_in_green = False
                if all_next_phase_signals_in_green:
                    # Transition complete
                    phase_changing = False
                    current_phase = next_phase
            else:
                # Wait for current phase signals to turn red
                pass
        else:
            # Not changing phase, keep signals as they are
            pass

        # Prepare the prediction to send
        prediction = {'signals': []}
        for signal_group, signal_info in signal_timers.items():
            prediction['signals'].append({'name': signal_group, 'state': signal_info['state']})

        # Update the desired phase of the traffic lights
        next_signals = {}
        current_tick = state.simulation_ticks

        for signal in prediction['signals']:
            actions[current_tick] = (signal['name'], signal['state'])
            next_signals[signal['name']] = signal['state']

        signal_logic_errors = input_queue.put(next_signals)

        if signal_logic_errors:
            errors.append(signal_logic_errors)

    # End of simulation, return the score

    # Transform the score to the range [0, 1]
    if state.total_score == 0:
        state.total_score = 1e9

    inverted_score = 1. / state.total_score
    

    return inverted_score

if __name__ == '__main__':
    run_game()