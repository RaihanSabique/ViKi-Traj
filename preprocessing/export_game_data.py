"""Export the trials of the reaching game from the raw game exports to one MATLAB file per participant.

A raw export <pid>.csv has one row per session and game (name, user_id, age, location, study_day, game_type, and
the trials of the session in 'data'). The trials of the reaching game, called paths by the game, are written as
struct 'Reaching' (one field per column, one row per trial, and 'Properties') to <out>/<pid>.mat, keeping the
other variables of an existing file. alignment/vicon/run_alignment.m and dataset/ read these files.

    python preprocessing/export_game_data.py [--raw data/game_raw] [--out data/game_data]
"""
import argparse
import ast
import math
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
GAME_TYPE = "Reaching"      # game_type of the reaching game in the raw export
TABLE = "Reaching"          # variable name in <pid>.mat

# Columns kept, in this order, and their names in the MATLAB table
column_selected = ['pid', 'user_id', 'age', 'study_location', 'study_day', 'trial_hand', 'trial_no',
                   'trial_start_time', 'trial_end_time', 'road_path', 'target_position', 'rocket_position',
                   'rocket_touch_begin_position', 'rocket_touch_begin_time',
                   'rocket_touch_lift_position', 'rocket_touch_lift_time',
                   'rocket_movement_path', 'rocket_movement_time', 'rocket_movement_time_step_ms',
                   'rocket_movement_time_step_sec', 'time_diff(sec)', 'rocket_movement_dist(px)',
                   'other_touch_begin_position', 'other_touch_begin_time', 'other_touch_begin_rocket_position',
                   'other_touch_lift_position', 'other_touch_lift_time', 'other_touch_lift_rocket_position',
                   'other_touch_movement_path', 'other_touch_movement_time', 'other_touch_movement_rocket_position']
column_renamed = ['pid', 'user_id', 'age', 'study_location', 'study_day', 'trial_hand', 'trial_no',
                  'trial_start_time', 'trial_end_time', 'road_path', 'target_pos', 'rocket_pos',
                  'RT_begin_pos', 'RT_begin_time', 'RT_lift_pos', 'RT_lift_time',
                  'RMov_path', 'RMov_time', 'RMov_time_step_ms', 'RMov_time_step_sec', 'time_diff_s', 'RMov_dist_px',
                  'OT_begin_pos', 'OT_begin_time', 'OT_begin_rocket_pos',
                  'OT_lift_pos', 'OT_lift_time', 'OT_lift_rocket_pos',
                  'OTMov_path', 'OTMov_time', 'OTMov_rocket_pos']


def time_diff(t1, t2):
    """Seconds from t1 to t2."""
    a = datetime.strptime(t1, '%Y-%m-%d %H:%M:%S.%f')
    b = datetime.strptime(t2, '%Y-%m-%d %H:%M:%S.%f')
    return (b - a).total_seconds()


def calculate_age(born):
    """Age in years today from a date of birth 'YYYY-MM-DD'; a number is returned as int."""
    if not isinstance(born, str):
        return int(born)
    born = datetime.strptime(born, '%Y-%m-%d')
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def total_movement_distance(points):
    """Length of the polyline through the (x, y) points."""
    if len(points) < 2:
        return 0.0
    total_distance = 0.0
    for i in range(1, len(points)):
        total_distance += euclidean_distance(points[i - 1], points[i])
    return total_distance


def python_time_to_milliseconds(trial_start_time, time_array):
    """Milliseconds from trial_start_time to each timestamp."""
    python_times = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") for ts in time_array]
    start_time = datetime.strptime(trial_start_time, '%Y-%m-%d %H:%M:%S.%f')
    return [(t - start_time).total_seconds() * 1000 for t in python_times]


def python_time_to_seconds(trial_start_time, time_array):
    """Seconds from trial_start_time to each timestamp."""
    python_times = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") for ts in time_array]
    start_time = datetime.strptime(trial_start_time, '%Y-%m-%d %H:%M:%S.%f')
    return [(t - start_time).total_seconds() for t in python_times]


def convert_timestring_to_array(time_string):
    if len(time_string) < 1:
        return []
    return [t.strip() for t in time_string]


def extract_movement_data(data):
    """Trials of one session from the 'data' string '[[key: value, ...], [key: value, ...]]', one row per trial."""
    movement = []
    for trial in data[2:-2].split("], ["):
        movement.append(eval("{" + trial + "}"))      # the fields of one trial as a dict literal
    return pd.DataFrame(movement)


def append_df_as_matlab_table(df, filename, table_name='T'):
    """Save df as struct table_name in filename (cell arrays for list columns, numeric columns as vectors, and
    'Properties' with variable names, descriptions, units and types), keeping the other variables of the file."""
    variable_descriptions = {
        # participant
        'pid': 'Unique identifier for each study participant',
        'user_id': 'User identification number or code',
        'dob': 'Date of birth of the participant',
        'age': 'Age of the participant',
        'study_location': 'Location where the study was conducted',
        'study_day': 'Day on which the study session took place',

        # trial
        'trial_hand': 'Which hand was used for the trial (left/right)',
        'trial_no': 'Sequential trial number within the session',
        'trial_start_time': 'Timestamp marking the beginning of the trial',
        'trial_end_time': 'Timestamp marking the end of the trial',
        'road_path': 'Path or route configuration for the trial',
        'target_pos': 'Position coordinates of the target',
        'rocket_pos': 'Position coordinates of the game object',

        # RT_*: touches on the game object
        'RT_begin_pos': 'Coordinates where game-object touch began',
        'RT_begin_time': 'Timestamp when game-object touch began',
        'RT_lift_pos': 'Coordinates where game-object touch ended',
        'RT_lift_time': 'Timestamp when game-object touch ended',

        # RMov_*: movement of the game object
        'RMov_path': 'Complete path trajectory of game-object movement',
        'RMov_time': 'Time series data for game-object movement',
        'RMov_time_step_ms': 'Milliseconds from trial_start_time to each RMov_time sample',
        'RMov_time_step_sec': 'Seconds from trial_start_time to each RMov_time sample',
        'time_diff_s': 'Trial duration logged by the game (trial_end_time - trial_start_time) in seconds',
        'RMov_dist_px': 'Total distance traveled in pixels for game-object movement based on the path',

        # OT_*: touches elsewhere on the screen
        'OT_begin_pos': 'Starting coordinates of touches off the game object',
        'OT_begin_time': 'Timestamp when other touch began',
        'OT_begin_rocket_pos': 'Game-object position when other touch began',
        'OT_lift_pos': 'Ending coordinates of touches off the game object',
        'OT_lift_time': 'Timestamp when other touch ended',
        'OT_lift_rocket_pos': 'Game-object position when other touch ended',

        # OTMov_*: movement of the other touches
        'OTMov_path': 'Complete path trajectory of other touch movements',
        'OTMov_time': 'Time series data for other touch movements',
        'OTMov_rocket_pos': 'Game-object positions corresponding to other touch movements'
    }

    new_table_data = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_value = df[col].iloc[0] if len(df[col]) > 0 else None
            if isinstance(sample_value, (list, np.ndarray)):
                # cell array of numeric arrays
                cell_array = np.empty(len(df[col]), dtype=object)
                for i, item in enumerate(df[col].values):
                    if isinstance(item, str):
                        try:
                            if item.startswith('[') and item.endswith(']'):
                                parsed_array = ast.literal_eval(item)
                                cell_array[i] = np.array(parsed_array)
                            else:
                                # numbers separated by spaces or commas
                                parsed_array = [float(x) for x in item.replace(',', ' ').split() if x.strip()]
                                cell_array[i] = np.array(parsed_array)
                        except Exception:
                            cell_array[i] = str(item)
                    elif isinstance(item, (list, tuple)):
                        cell_array[i] = np.array(item)
                    elif isinstance(item, np.ndarray):
                        cell_array[i] = item
                    else:
                        cell_array[i] = str(item)
                new_table_data[col] = cell_array.reshape(-1, 1)
            else:
                # cell array of strings
                new_table_data[col] = np.array([[str(x)] for x in df[col].values], dtype=object)
        else:
            new_table_data[col] = df[col].values.reshape(-1, 1)

    properties = {}
    properties['VariableNames'] = np.array([col for col in df.columns], dtype=object)
    descriptions = []
    for col in df.columns:
        desc = variable_descriptions.get(col, f'Variable: {col}')
        descriptions.append(desc)
    properties['VariableDescriptions'] = np.array(descriptions, dtype=object)

    # units from the column names
    units = []
    for col in df.columns:
        if 'id' in col.lower():
            units.append('identifier')
        elif col in ['dob']:
            units.append('date')
        elif col in ['age']:
            units.append('years')
        elif 'time' in col.lower() and 'step' not in col.lower() and 'diff' not in col.lower():
            units.append('timestamp')
        elif 'time_step_ms' in col.lower():
            units.append('milliseconds')
        elif 'time_diff' in col.lower():
            units.append('seconds')
        elif 'dist' in col.lower() and 'px' in col.lower():
            units.append('pixels')
        elif 'pos' in col.lower() or 'path' in col.lower():
            units.append('coordinates')
        elif col in ['trial_hand', 'study_location', 'road_path']:
            units.append('categorical')
        elif col in ['trial_no', 'study_day']:
            units.append('numeric')
        else:
            units.append('mixed')
    properties['VariableUnits'] = np.array(units, dtype=object)

    data_types = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_value = df[col].iloc[0] if len(df[col]) > 0 else None
            if isinstance(sample_value, (list, np.ndarray)):
                data_types.append('cell_array')
            else:
                data_types.append('string')
        elif np.issubdtype(df[col].dtype, np.integer):
            data_types.append('integer')
        elif np.issubdtype(df[col].dtype, np.floating):
            data_types.append('double')
        else:
            data_types.append('mixed')
    properties['VariableTypes'] = np.array(data_types, dtype=object)

    properties['TableMetadata'] = {
        'Description': 'Touch interaction and game-object movement data of the reaching game',
        'GameType': 'Reaching',
        'NumObservations': len(df),
        'NumVariables': len(df.columns),
        'CreatedDate': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    new_table_data['Properties'] = properties

    if os.path.exists(filename):
        try:
            existing_data = loadmat(filename)
            existing_data = {k: v for k, v in existing_data.items() if not k.startswith('__')}
            existing_data[table_name] = new_table_data
            savemat(filename, existing_data, oned_as='column')
        except Exception as e:
            print(f"{filename}: cannot update ({e}); writing a new file with {table_name} only")
            savemat(filename, {table_name: new_table_data}, oned_as='column')
    else:
        savemat(filename, {table_name: new_table_data}, oned_as='column')


def read_csv_files_from_directory(directory_path):
    """Yield (file name, DataFrame) for every CSV file of a folder."""
    directory = Path(directory_path)
    if not directory.exists():
        raise ValueError(f"Directory {directory_path} does not exist")
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            yield csv_file.name, df
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")


def individual_process_and_export_to_matlab(df, participant_name, out_dir):
    """Trials of every session in df (rows of one participant's export) -> TABLE in <out_dir>/<participant>.mat."""
    trials = pd.DataFrame()
    for _, row in df.iterrows():
        d = row["data"]
        day = row["study_day"]
        participant = row["name"]
        user_id = row["user_id"]
        age = row["age"]
        location = row["location"]
        data = extract_movement_data(d)
        data["pid"] = data.apply(lambda x: participant, axis=1)
        data["user_id"] = data.apply(lambda x: user_id, axis=1)
        data["age"] = data.apply(lambda x: calculate_age(age), axis=1)
        data["study_location"] = data.apply(lambda x: location, axis=1)
        data["study_day"] = data.apply(lambda x: day, axis=1)
        data['target_position'] = data['target_position'].apply(lambda x: [x])
        data['rocket_position'] = data['rocket_position'].apply(lambda x: [x])
        data["rocket_movement_time"] = data["rocket_movement_time"].apply(lambda x: convert_timestring_to_array(x))
        data['rocket_movement_time_step_ms'] = data.apply(
            lambda x: python_time_to_milliseconds(x.trial_start_time, x.rocket_movement_time), axis=1)
        data['rocket_movement_time_step_sec'] = data.apply(
            lambda x: python_time_to_seconds(x.trial_start_time, x.rocket_movement_time), axis=1)
        data["time_diff(sec)"] = data.apply(lambda x: time_diff(x.trial_start_time, x.trial_end_time), axis=1)
        data["rocket_movement_dist(px)"] = data.apply(lambda x: total_movement_distance(x.rocket_movement_path), axis=1)
        trial_df = data[column_selected]
        trials = pd.concat([trials, trial_df], ignore_index=True)
    trials.columns = column_renamed
    filename = str(Path(out_dir) / f"{participant_name}.mat")
    append_df_as_matlab_table(trials, filename, TABLE)
    print(f"{participant_name}: {len(trials)} trials -> {filename}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "game_raw",
                    help="folder of the raw game exports, one <pid>.csv per participant")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "game_data", help="folder of the <pid>.mat files")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, dataframe in read_csv_files_from_directory(args.raw):
        participant = filename.split(".")[0]
        df = dataframe[dataframe['game_type'] == GAME_TYPE]
        individual_process_and_export_to_matlab(df, participant, args.out)


if __name__ == "__main__":
    main()
