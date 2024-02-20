

<img src="https://github.com/arthurbirate/cough_detection_visi_age/blob/main/cough_detection_4.0/logo/caringminds.jpg" alt="Alt text" width="350"/>


## cough detection : Visi-Age


# 1. Project Overview
The Cough Detection System is designed to monitor the health of elderly individuals by utilizing XGBoost, a machine learning algorithm, to detect coughs. Coughing can be an indicator of various health conditions, and this system aims to provide an early warning mechanism for caregivers and healthcare professionals.

# 2. Project Goals

### Continuous Monitoring: 
The system aims to provide continuous monitoring of coughing episodes, enabling caregivers to track the severity of coughs over time. This longitudinal data can offer valuable insights into the individual's health condition and progression of any underlying illnesses.


### Remote Monitoring: 
With the capability to integrate with remote monitoring systems and communication platforms, the Cough Detection System facilitates remote caregiving. This feature is particularly beneficial for elderly individuals who live alone or in remote locations, allowing caregivers to monitor their health status remotely and provide assistance as needed.

# 3. Key Components:

### Audio Input Module:  
The system utilizes microphones or audio input devices to capture ambient sounds, including coughs. Python libraries such as PyAudio or SoundDevice are employed to access audio input streams from microphones.

### Cough Detection Algorithm:

Machine learning algorithms, such as XGBoost, are employed to analyze audio data and detect cough sounds accurately. These algorithms are trained on labeled audio samples to distinguish coughs from other noises effectively.


### Data Logging and Analysis:

The system records coughing events along with relevant metadata, such as timestamp and possibiblty, for further analysis. Data logging allows caregivers to review historical coughing patterns and identify trends or anomalies in the individual's health condition. and the data is sent to the application where caregivers can view through a visualization how msny times a patient has been coughing in a week or monthly 

# 4.Prerequisites


Before running the project, ensure you have the following prerequisites installed:

- Python (version 3.11 and above)
- Dependencies listed in `requirements.txt` (install using `pip install -r requirements.txt`)
- Audio Input Device Configuration

### Audio Input Device Configuration

Before running the project, you may need to configure the audio input device to use for capturing cough sounds. To assist in selecting the appropriate audio device, you can run the following Python script:

```python
import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    print("Available audio devices:")
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        device_name = device_info.get('name')
        device_id = device_info.get('index')
        print(f"Device {i}: {device_name}, ID: {device_id}")

    p.terminate()

if __name__ == "__main__":
    list_audio_devices()


This script lists the available audio devices along with their IDs. Run the script to identify the audio input device you want to use for capturing cough sounds. Note down the device ID for configuration in the main project script.

Ensure that the selected audio device is properly connected and configured on your system before running the project.

