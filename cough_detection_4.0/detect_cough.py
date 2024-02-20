# import time
# import pickle
# import os
# import sys
# import numpy as np
# import sounddevice as sd
# import threading
# import requests
# from datetime import datetime
#
# from scipy.io.wavfile import write
#
# from src.DSP import classify_cough
#
# sys.path.append(os.path.abspath('./src'))
# sys.path.append(os.path.abspath('./src/cough_detection'))
#
# THRESHOLD = 45
# RECORDING_INTERVAL = 1  # Time in seconds between recordings
#
#
# url = 'https://appserviceproject4tm20241.azurewebsites.net/api/visiage/cough'
# microphone_ID_1 = 2
# microphone_ID_2 = 0
# def record_cough_thread(duration, fs, model, scaler, stop_event):
#     frames = []
#
#     def callback(indata, frames_list, time, status):
#         if status:
#             print(f"Error in audio input: {status}")
#             return
#         frames_list.append(indata.copy())
#
#     with sd.InputStream(channels=1, callback=lambda indata, frames_list, time, status: callback(indata, frames, time, status),
#                         blocksize=int(fs * duration), dtype=np.int16,device=device):
#         while not stop_event.is_set():
#             time.sleep(RECORDING_INTERVAL)
#
#             if len(frames) > 0:
#                 audio_data = np.concatenate(frames)
#                 frames.clear()
#
#                 prob = classify_cough(audio_data, fs, model, scaler)
#                 convert_prob = int(prob * 100 )
#
#                 if convert_prob >= THRESHOLD:
#                     print(f"Cough detected, Severity: {convert_prob}%")
#
#                     current_datetime = datetime.now()
#                     formatted_timestamp = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")
#
#                     data = {
#                         "microphoneId": str(1),
#                         "timeStamp": formatted_timestamp,
#                         "severity": convert_prob
#                     }
#
#                     try:
#
#                         response = requests.post(url, json=data)
#
#                         response.raise_for_status()  # Raise an HTTPError for bad responses
#                         if response.status_code == 201:
#                             print(f" Data successfully sent : {response.status_code}")
#                         else:
#                             print("Error Sending Data.??")
#                     except requests.exceptions.RequestException as e:
#                         print(f"Error sending data: {e}")
#                 else:
#                     print("No cough detected")
#
# def main():
#     model = pickle.load(open(os.path.join('./models', 'cough_classifier'), 'rb'))
#     scaler = pickle.load(open(os.path.join('./models', 'cough_classification_scaler'), 'rb'))
#
#     fs = 47100  # Sample rate
#     duration = 8  # Duration of each recording in seconds
#
#     stop_event = threading.Event()
#     recording_thread = threading.Thread(target=record_cough_thread, args=(duration, fs, model, scaler, stop_event))
#
#     try:
#         recording_thread.start()
#         print("Listening for coughs in real-time... Press Ctrl+C to stop.")
#         recording_thread.join()
#     except KeyboardInterrupt:
#         print("\nStopping the program.")
#         stop_event.set()  # Signal the recording thread to stop
#         recording_thread.join()  # Wait for the recording thread to finish
#
# if __name__ == '__main__':
#     main()
#


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

THRESHOLD = 60
RECORDING_INTERVAL = 18  # Time in seconds between recordings

url = 'https://appserviceproject4tm20241.azurewebsites.net/api/visiage/cough'

microphone_ID_1 = 2
microphone_ID_2 = 0


def record_cough_thread(device, duration, fs, model, scaler, stop_event):
    frames = []

    def callback(indata, frames_list, time, status):
        if status:
            print(f"Error in audio input: {status}")
            return
        frames_list.append(indata.copy())

    with sd.InputStream(channels=1,
                        callback=lambda indata, frames_list, time, status: callback(indata, frames, time, status),
                        blocksize=int(fs * duration), dtype=np.int16, device=device):
        while not stop_event.is_set():
            start_time = time.time()  # Record start time

            time.sleep(RECORDING_INTERVAL)

            if len(frames) > 0:
                audio_data = np.concatenate(frames)
                frames.clear()

                prob = classify_cough(audio_data, fs, model, scaler)
                convert_prob = int(prob * 100)

                current_datetime = datetime.now()
                formatted_timestamp = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")

                if convert_prob >= THRESHOLD:
                    processing_time = time.time() - start_time
                    print(
                        f"{formatted_timestamp}: Cough detected on Microphone {device}, probability: {convert_prob}%, Processing Time: {processing_time:.2f} seconds")

                    data = {
                        "microphoneId": str(device),
                        "timeStamp": formatted_timestamp,
                        "severity": convert_prob
                    }

                    try:
                        response = requests.post(url, json=data)
                        response.raise_for_status()  # Raise an HTTPError for bad responses
                        if response.status_code == 201:
                            print(f"{formatted_timestamp}: Data successfully sent: {response.status_code}")
                        else:
                            print(f"{formatted_timestamp}: Error Sending Data.??")
                    except requests.exceptions.RequestException as e:
                        print(f"{formatted_timestamp}: Error sending data: {e}")
                else:
                    print(f"{formatted_timestamp}: No cough detected on Microphone {device}")


def main():
    model = pickle.load(open(os.path.join('./models', 'cough_classifier'), 'rb'))
    scaler = pickle.load(open(os.path.join('./models', 'cough_classification_scaler'), 'rb'))

    fs = 48000  # Sample rate
    duration = 10  # Duration of each recording in seconds

    stop_event = threading.Event()

    # Create threads for each microphone
    thread_1 = threading.Thread(target=record_cough_thread,
                                args=(microphone_ID_1, duration, fs, model, scaler, stop_event))
    thread_2 = threading.Thread(target=record_cough_thread,
                                args=(microphone_ID_2, duration, fs, model, scaler, stop_event))

    try:
        thread_1.start()
        thread_2.start()

        print("Listening for coughs in real-time... Press Ctrl+C to stop.")

        while not stop_event.is_set():
            time.sleep(1)  # Check every 1 second for interruption or exit condition

    except KeyboardInterrupt:
        print("\nStopping the program.")
        stop_event.set()  # Signal the recording threads to stop
        thread_1.join()  # Wait for the recording threads to finish
        thread_2.join()


if __name__ == '__main__':
    main()
