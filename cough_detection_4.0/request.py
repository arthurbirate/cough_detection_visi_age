# ### COUGH POST REQUEST ###

import requests

# Set the request url
url = 'https://appserviceproject4tm20241.azurewebsites.net/api/visiage/cough'
# Set the request headers
data = {severity: "70", timeStamp: "2021-10-18T12:00:00Z", amount: 10, cameraRoomId: 1}

# Make a POST request
response = requests.post(url, json=data)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    print('POST request successful')
    print('Response:', response.text)
else:
    print(f'Error: {response.status_code}')
    print('Response:', response.text)





import time
import pickle
import os
import sys
import numpy as np
import sounddevice as sd
import threading
import requests
from datetime import datetime

from scipy.io.wavfile import write
from src.DSP import classify_cough

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('./src/cough_detection'))

THRESHOLD = 45
RECORDING_INTERVAL = 2  # Time in seconds between recordings

# Replace this with your FastAPI server endpoint
FASTAPI_ENDPOINT = "http://YOUR_FASTAPI_SERVER:port/endpoint"

def record_cough_thread(duration, fs, model, scaler, stop_event):
    frames = []

    def callback(indata, frames_list, time, status):
        if status:
            print(f"Error in audio input: {status}")
            return
        frames_list.append(indata.copy())

    with sd.InputStream(channels=1, callback=lambda indata, frames_list, time, status: callback(indata, frames, time, status),
                        blocksize=int(fs * duration), dtype=np.int16):
        while not stop_event.is_set():
            time.sleep(RECORDING_INTERVAL)

            if len(frames) > 0:
                audio_data = np.concatenate(frames)
                frames.clear()

                prob = classify_cough(audio_data, fs, model, scaler)
                convert_prob = prob * 100

                if convert_prob >= THRESHOLD:
                    print(f"Cough detected, Probability: {convert_prob}%")

                    # Send data to FastAPI
                    current_datetime = datetime.now()
                    date_str = current_datetime.strftime("%Y-%m-%d")
                    time_str = current_datetime.strftime("%H:%M:%S")
                    
                    data = {
                        "date": date_str,
                        "time": time_str,
                        "probability": convert_prob
                    }

                    try:
                        response = requests.post(FASTAPI_ENDPOINT, json=data, timeout=5)
                        response.raise_for_status()  # Raise an HTTPError for bad responses
                        if response.status_code == 200:
                            print("Data sent to FastAPI.")
                        else:
                            print(f"Failed to send data to FastAPI. Status code: {response.status_code}")
                    except requests.exceptions.RequestException as e:
                        print(f"Error sending data to FastAPI: {e}")

                else:
                    print("No cough detected")

def main():
    model = pickle.load(open(os.path.join('./models', 'cough_classifier'), 'rb'))
    scaler = pickle.load(open(os.path.join('./models', 'cough_classification_scaler'), 'rb'))

    fs = 47100  # Sample rate
    duration = 10  # Duration of each recording in seconds

    stop_event = threading.Event()
    recording_thread = threading.Thread(target=record_cough_thread, args=(duration, fs, model, scaler, stop_event))

    try:
        recording_thread.start()
        print("Listening for coughs in real-time... Press Ctrl+C to stop.")
        recording_thread.join()
    except KeyboardInterrupt:
        print("\nStopping the program.")
        stop_event.set()  # Signal the recording thread to stop
        recording_thread.join()  # Wait for the recording thread to finish

if __name__ == '__main__':
    main()
