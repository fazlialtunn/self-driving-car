# 🛣️ Self-Driving Car with Deep Learning (NVIDIA Model)

This project is a deep learning-based self-driving car system built using Python, Keras, Flask, and OpenCV. The model mimics the NVIDIA self-driving car architecture and predicts steering angles from real-time camera images.

## 🚗 Project Overview

This project includes:

- Data preprocessing and augmentation (pan, zoom, brightness, flip)
- Model training using NVIDIA's CNN architecture
- Real-time inference via a Flask server with SocketIO
- Use of behavioral cloning to teach a model how to drive

## 📁 Project Structure

```
self-driving-car/
├── model/
│   └── model.h5                # Trained deep learning model
├── track/                      # Driving dataset (downloaded from Udacity or similar)
│   ├── IMG/                    # Images captured during driving
│   └── driving_log.csv         # Steering and throttle data
├── app.py                      # Real-time inference server
├── train.py                    # Data loading, augmentation, and model training
└── README.md
```

## 📦 Installation

1. Clone the repo:

    ```bash
    git clone https://github.com/fazlialtunn/self-driving-car.git
    cd self-driving-car
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Example dependencies:

- tensorflow  
- keras  
- opencv-python  
- matplotlib  
- pandas  
- Flask  
- python-socketio  
- eventlet  
- imgaug  

## 📥 Data Collection

You can clone a public dataset:

```bash
git clone https://github.com/rslim087a/track

Make sure driving_log.csv and the IMG/ folder are in the track/ directory.

🧪 Training the Model

Run the training script to preprocess data, apply augmentation, and train the model:

python train.py

Key features:
	•	Balanced steering angle histogram
	•	Data augmentation (pan, zoom, brightness, flip)
	•	NVIDIA model architecture
	•	Trained using MSE loss and Adam optimizer

You can customize the batch size, number of epochs, and augmentation probabilities.

📈 Training Results
	•	Validation Loss stabilized around 0.03
	•	Plots of training and validation loss available

🚀 Run the Inference Server

After training the model:

python app.py

This starts a Flask-SocketIO server on port 4567, ready to receive telemetry data (image, speed) from the simulator and return steering and throttle predictions.

🧠 Model Architecture (NVIDIA)

model = Sequential()
model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
model.add(Conv2D(64, (5, 5), activation='elu'))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

💡 Highlights
	•	Uses real-time image data from the simulator to control steering
	•	Applies real-world inspired augmentations to improve generalization
	•	Implements NVIDIA’s self-driving car CNN design
	•	Designed to be deployable on Jetson, Raspberry Pi, or real vehicles

📷 Sample Augmentations

Zoom	Pan	Brightness	Flip
✅	✅	✅	✅

📤 Export Model

After training:

model.save('model.h5')

To download in Colab:

from google.colab import files
files.download('model.h5')


⸻

🏁 Acknowledgements
	•	NVIDIA for the original model architecture
	•	Udacity for open datasets and the self-driving simulator
	•	Community resources on behavioral cloning

⸻

🛠️ Future Improvements
	•	Add lateral and longitudinal control with PID
	•	Use RNNs or transformers for temporal modeling
	•	Deploy on edge devices

⸻