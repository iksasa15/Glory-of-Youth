# Glory of Youth - Football Player Analysis and Training System

## 📌 Overview
The "Glory of Youth" project is an integrated system combining:
- AI-powered video analysis
- 3D training simulations
- Web dashboards for coaches and players

## ✨ Key Features

### 🖥️ Performance Analysis
- Precise player movement tracking using OpenCV and MediaPipe
- Player performance metrics (passes, dribbles, speed)
- Textual data analysis from coaches' reports

### 🎮 Training Simulation
- 3D scenes for correct movement simulations
- Interactive training model (as shown in correct_movements.mp4)
- Unity engine for displaying correct movements
- C# programming (in Assembly-CSharp.csproj and DSS.sln files)

### 🌐 Web Interfaces
- `coach_dribbling.html`: Coach's dashboard
- `home_dribbling.html`: Player's monitoring panel

## ⚙️ Technologies Used

### Analysis Module (Python)
- OpenCV for video processing
- MediaPipe for movement analysis
- NumPy for data processing

### Web Module (HTML/JS)
- HTML/CSS for basic structure
- JavaScript for interactivity

##  How to Run

### Analysis Module
```bash
pip install opencv-python mediapipe numpy
python app.py
