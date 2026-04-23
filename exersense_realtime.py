import serial
import time
import datetime
import numpy as np
import os
from scipy import interpolate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
port = 'COM6'
baud = 9600

warmup_reps_count = 15
target_length = 50  # Number of samples per rep after resampling

rom_threshold = 15000  # Minimum range of motion to consider it a valid rep
movement_threshold = 5000  # Minimum movement to consider it a valid rep
return_threshold = 3000  # Y value indicating arm return position to start position

# Warmup quality thresholds
min_samples = 30  # Reject reps that are too fast
max_samples = 80  # Reject reps that are too slow
min_rom = 15000  # Reject shallow reps


#####################################################################
# setting up folders
#####################################################################

def setup_folders(exercise_name):
    base_folder = os.path.join('exercises', exercise_name)

    folders = {
        'base': base_folder,
        'warmup': os.path.join(base_folder, 'warmup'),
        'model': os.path.join(base_folder, 'model'),
        'sessions': os.path.join(base_folder, 'sessions')
    }

    for f in folders.values():
        os.makedirs(f, exist_ok=True)

    return folders


#####################################################################
# Arduino connection
#####################################################################

def connect_arduino(port, baud):
    ser = serial.Serial(port, baud)
    time.sleep(3)  # Wait for connection to stabilize
    print("Connected to Arduino.")
    return ser


##################################################################
# calibrating the arm down position by averaging Y value when the arm is relaxed over a few seconds
##################################################################

def calibrate_arm_down(ser, calibration_time=3):
    print("Calibrating arm down position. Keep your arm relaxed and down...")
    time.sleep(1)
    calibration_values = [] #list since i dont know amount of data to be recorded here bruh
    start_time = time.time()

    while time.time() - start_time < calibration_time: #keep checking until 3 seconds have passed
        line = ser.readline().decode().strip() #read a line, turn bytes into string, clean up whitespace

        if line.startswith("DATA,"):
            parts = line[5:].split(",") # skip data prefix and break into parts at each comma
            forearm_y = int(parts[1])  # Changed from parts[0] to parts[1]
            calibration_values.append(forearm_y)

    if len(calibration_values) == 0:
        print("ERROR: No data received from Arduino!")
        return None

    arm_down_y = sum(calibration_values) // len(calibration_values) #gets the average which is used as the arm down value
    print(f"Calibration done. Arm down Y = {arm_down_y}")

    with open('state.json', 'w') as f:
        json.dump({
            "score": 0,
            "rom": 0,
            "status": "calibrated"
        }, f)

    return arm_down_y


def detect_rep(ser, arm_down_y):
    rep_data = []
    min_y = 99999
    max_y = -99999
    collecting = False #movement detected flag

    while True:
        if os.path.exists('stop_set.flag'): #has stop set been clicked
            os.remove('stop_set.flag')
            return None, None, None, None

        line = ser.readline().decode().strip()

        if not line.startswith("DATA,"):
            continue

        data = line[5:]
        parts = data.split(",")
        y = int(parts[1])  # Changed from parts[0] to parts[1] becuase 0 is forearmX and 1 is forearmY

        # Track min/max Y during this rep
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        rom = max_y - min_y

        # Start collecting when movement detected
        if not collecting and rom > movement_threshold:
            collecting = True
            print(f"  Movement detected, ROM = {rom}")

        if collecting:
            rep_data.append(data)

        # Rep complete: good ROM and current y is close to the resting position
        if collecting and rom > rom_threshold and abs(y - arm_down_y) < return_threshold:
            time.sleep(0.5)  # Prevent double-count
            return rep_data, rom, min_y, max_y

    return None, None, None, None

##################################################################
# saving the reps
##################################################################

def save_rep(rep_data, folder, rep_count):
    filename = os.path.join(folder, f"rep_{rep_count:03d}.csv")
    with open(filename, 'w') as f:
        f.write("forearmX,forearmY,forearmZ,elbowDriftX,elbowDriftY,elbowDriftZ,timestamp\n")
        for row in rep_data:
            f.write(row + "\n")
    return filename


##################################################################
# preprocessing same as load_reps.py
##################################################################

def convert_reps(rep_data):
    data = []
    for line in rep_data:
        parts = line.strip().split(',')
        row = [int(parts[i]) for i in range(6)]
        data.append(row)
    rep_array = np.array(data)
    return rep_array


def resample_rep(rep_array):
    original_length = rep_array.shape[0] #numberf of rows
    num_features = rep_array.shape[1] #number of columns

    original_time = np.linspace(0, 1, original_length) #evenly  spaced numbers between 0 and 1, length of original data
    target_time = np.linspace(0, 1, target_length)

    resampled_rep = np.zeros((target_length, num_features))

    for feature in range(num_features):
        interpolation = interpolate.interp1d(original_time, rep_array[:, feature], kind='linear')
        # interp1d creates a linear interpolation function based on the original time points and the feature values so that we can estimate feature values at new time points
        resampled_rep[:, feature] = interpolation(target_time)
        # we then use this function to get the feature values at the new target time points

    return resampled_rep


def normalize_rep(rep_array, global_min, global_max):
    normalized = np.zeros_like(rep_array, dtype=float)

    for feature in range(rep_array.shape[1]):
        range_val = global_max[feature] - global_min[feature]
        if range_val > 0:
            # normalization formula to scale values between 0 and 1
            # same formula as (value - min) / (max - min)
            normalized[:, feature] = (rep_array[:, feature] - global_min[feature]) / range_val

    return normalized


def preprocess_rep(rep_data, global_min, global_max):
    rep_array = convert_reps(rep_data)  # converts the rep data to a numpy array
    resampled = resample_rep(rep_array)  # resamples the rep to the target length
    normalized = normalize_rep(resampled, global_min, global_max)  # normalizes the rep using global min and max values
    flattened = normalized.reshape(
        (1, -1))  # reshapes the normalized rep to a single row. 1,-1 means 1 row and as many columns as needed
    return flattened



##################################################################
# autoencoder. same as autoencoder.py
##################################################################

def build_autoencoder(input_size):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_size,)),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_size, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

#change to increase the range for the threshold
def calculate_score(mse, threshold):
    if mse <= threshold:
        score = 50 + 50 * (1 - mse / threshold)
    else:
        score = 50 * np.exp(-(mse - threshold) / threshold)

    score = max(0, min(100, score))
    return score


##################################################################
# warmup phase with quality checks
##################################################################

def warmup_phase(ser, arm_down_y, folders):
    print(f"\n{'=' * 50}")
    print("WARMUP PHASE")
    print(f"{'=' * 50}")
    print(f"Perform {warmup_reps_count} reps with PERFECT FORM")
    print(f"Keep consistent tempo (not too fast, not too slow)\n")

    warmup_reps = []
    warmup_roms = []
    warmup_samples = []
    warmup_min_ys = []
    warmup_max_ys = []

    i = 0
    while i < warmup_reps_count:
        print(f"Waiting for rep {i + 1}/{warmup_reps_count}...")

        rep_data, rom, min_y, max_y = detect_rep(ser, arm_down_y)
        if rep_data is None:
            print("Warmup interrupted.")
            return None, None, None, None, None, None

        samples = len(rep_data)

        # Quality checks
        if samples < min_samples:
            print(f"  ✗ Too fast ({samples} samples) - do another rep")
            continue

        if samples > max_samples:
            print(f"  ✗ Too slow ({samples} samples) - do another rep")
            continue

        if rom < min_rom:
            print(f"  ✗ Too shallow (ROM: {rom}) - do another rep")
            continue

        # Rep passed quality checks
        save_rep(rep_data, folders['warmup'], i + 1)

        rep_array = convert_reps(rep_data)
        resampled = resample_rep(rep_array)
        warmup_reps.append(resampled)
        warmup_roms.append(rom)
        warmup_samples.append(samples)
        warmup_min_ys.append(min_y)
        warmup_max_ys.append(max_y)

        print(f"  Rep {i + 1} ✓  Samples: {samples}, ROM: {rom}")

        with open('state.json', 'w') as f:
            json.dump({
                "score": 0,
                "rom": 0,
                "status": f"warmup rep {i + 1} of {warmup_reps_count}"
            }, f)


        i += 1

    # Show consistency stats
    print(f"\n{'─' * 40}")
    print("Warmup Quality Check:")
    print(f"  Samples range: {min(warmup_samples)}-{max(warmup_samples)}")
    print(f"  ROM range: {min(warmup_roms)}-{max(warmup_roms)}")
    print(f"{'─' * 40}")

    # Stack and calculate global normalization params
    warmup_data_raw = np.stack(warmup_reps)
    global_min = warmup_data_raw.min(axis=(0, 1))  # axis makes it so we only get min per feature across all samples
    global_max = warmup_data_raw.max(axis=(0, 1))  # axis makes it so we only get max per feature across all samples

    # Normalize and flatten
    warmup_data_normalized = np.zeros_like(warmup_data_raw, dtype=float)
    for i in range(warmup_data_raw.shape[0]):
        warmup_data_normalized[i] = normalize_rep(warmup_data_raw[i], global_min, global_max)

    warmup_data_flattened = warmup_data_normalized.reshape(
        (warmup_reps_count, -1))  # flattened version of the normalized warmup reps

    print(f"\nWarmup complete! Average ROM: {np.mean(warmup_roms):.0f}")

    warmup_min_y = np.mean(warmup_min_ys)
    warmup_max_y = np.mean(warmup_max_ys)

    print(f"Warmup min Y: {warmup_min_y:.0f}")
    print(f"Warmup max Y: {warmup_max_y:.0f}")
    

    return warmup_data_flattened, global_min, global_max, warmup_roms, warmup_min_y, warmup_max_y


##################################################################
# working set phase
##################################################################

def working_set_phase(ser, arm_down_y, session_folder, model, global_min, global_max, threshold, warmup_avg_rom, warmup_min_y, warmup_max_y):
    print(f"\n{'=' * 50}")
    print("WORKING SET")
    print(f"{'=' * 50}")
    print(f"Threshold: {threshold:.6f}")
    print("Perform your set. Press Ctrl+C to finish.\n")

    results = []
    rep_count = 0

    while True:
        if os.path.exists('stop_set.flag'):
            os.remove('stop_set.flag')
            break

        print(f"Waiting for rep {rep_count + 1}...")
        rep_data, rom, min_y, max_y = detect_rep(ser, arm_down_y)


        if rep_data is None:
            break

        rep_count += 1
        save_rep(rep_data, session_folder, rep_count)

        # Score the rep
        rep_flattened = preprocess_rep(rep_data, global_min, global_max)
        reconstruction = model.predict(rep_flattened, verbose=0)
        mse = np.mean((rep_flattened - reconstruction) ** 2)
        score = calculate_score(mse, threshold)

        warmup_range = warmup_max_y - warmup_min_y
        curl_proportion = (max_y - warmup_min_y) / warmup_range
        rom_percent = (curl_proportion ** 3) * 100
        rom_percent = max(0, min(120, rom_percent))

        # Status indicators
        if score >= 70:
            status = "Good"
        elif score >= 50:
            status = "OK"
        else:
            status = "Check form"

        print(f"  Rep {rep_count}: Score {score:.0f}  MSE: {mse:.6f}  ROM: {rom_percent:.0f}%  {status}")


        with open('state.json', 'w') as f:
            json.dump({
                "score": round(score),
                "rom": round(rom_percent),
                "status": status,
                "rep": rep_count
            }, f)


        results.append({'rep': rep_count, 'score': score, 'rom': rom, 'rom_percent': rom_percent, 'mse': mse})

    print("\nSet finished!")
    return results


##################################################################
# summary
##################################################################

def print_summary(warmup_roms, results):
    print(f"\n{'=' * 50}")
    print("SESSION SUMMARY")
    print(f"{'=' * 50}")

    if len(results) == 0:
        print("No working set reps recorded.")
        return

    warmup_avg_rom = np.mean(warmup_roms)
    scores = [r['score'] for r in results]
    roms = [r['rom'] for r in results]

    print(f"\nWarmup Avg ROM: {warmup_avg_rom:.0f}")
    print(f"Set Avg ROM:    {np.mean(roms):.0f} ({(np.mean(roms) / warmup_avg_rom) * 100:.0f}%)")
    print(f"\n{'─' * 40}")
    print("Rep Scores:")
    print(f"{'─' * 40}")

    for r in results:
        bar_len = int(r['score'] / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)

        flag = ""
        if r['score'] < 50:
            flag = " ⚠"
        elif r['rom_percent'] < 80:
            flag = " (low ROM)"

        print(f"Rep {r['rep']:2d}: {bar} {r['score']:.0f}{flag}")

    print(f"{'─' * 40}")
    print(f"Average Score: {np.mean(scores):.1f}")
    print(f"{'─' * 40}")

    with open('state.json', 'w') as f:
        json.dump({
            "score": round(np.mean(scores)),
            "rom": round((np.mean(roms) / warmup_avg_rom) * 100),
            "status": "complete",
            "results": results,
        }, f)

##################################################################
# main
##################################################################

def main():


    print("Exercises:")
    print("  1. Freeform Bicep Curl\n")
    exercise_name = "freeform_bicep_curl"

    folders = setup_folders(exercise_name)

    # Connect and calibrate
    ser = connect_arduino(port, baud)
    arm_down_y = calibrate_arm_down(ser)

    if arm_down_y is None:
        ser.close()
        return

    # Collect warmup
    warmup_data_flattened, global_min, global_max, warmup_roms, warmup_min_y, warmup_max_y = warmup_phase(ser, arm_down_y, folders)

    if warmup_data_flattened is None:
        ser.close()
        return


    # Train autoencoder
    print("\nTraining autoencoder...")
    model = build_autoencoder(warmup_data_flattened.shape[1])
    model.fit(warmup_data_flattened, warmup_data_flattened, epochs=100, batch_size=5, verbose=0)
    print("Training complete!")

    with open('state.json', 'w') as f:
        json.dump({
            "score": 0,
            "rom": 0,
            "status": "training complete"
        }, f)


    # Calculate threshold with debug output
    print("\nWarmup MSE values:")
    reconstructions = model.predict(warmup_data_flattened, verbose=0)
    errors = []
    for i in range(warmup_reps_count):
        mse = np.mean((warmup_data_flattened[i] - reconstructions[i]) ** 2)
        errors.append(mse)
        print(f"  Warmup Rep {i + 1} MSE: {mse:.6f}")

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    threshold = mean_error + 6 * std_error #################################################################change for being more lenient

    print(f"\nMean: {mean_error:.6f}, Std: {std_error:.6f}")
    print(f"Threshold (mean + 4*std): {threshold:.6f}")

    # Save model
    model.save(os.path.join(folders['model'], 'autoencoder.keras'))
    np.save(os.path.join(folders['model'], 'global_min.npy'), global_min)
    np.save(os.path.join(folders['model'], 'global_max.npy'), global_max)

    warmup_avg_rom = np.mean(warmup_roms)

    # Working set
    #input("\nPress ENTER to start working set...")

    print("\nPress 'Start Set' button on the website to begin...")
    # Clear any stale flag from a previous session
    if os.path.exists('start_set.flag'):
        os.remove('start_set.flag')

    print("\nPress 'Start Set' button on the website to begin...")
    while not os.path.exists('start_set.flag'):
        time.sleep(0.5)
    os.remove('start_set.flag')

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    session_folder = os.path.join(folders['sessions'], timestamp)
    os.makedirs(session_folder)

    results = working_set_phase(
        ser, arm_down_y, session_folder, model,
        global_min, global_max, threshold, warmup_avg_rom,
        warmup_min_y, warmup_max_y
    )

    print_summary(warmup_roms, results)

    ser.close()
    print("\nSession complete!")


if __name__ == "__main__":
    main()