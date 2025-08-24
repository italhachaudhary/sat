# AI-Powered Surgical Assistance Tool (SAT)

## Overview
The AI-Powered Surgical Assistance Tool (SAT) is an advanced application designed to assist surgeons and medical professionals by leveraging computer vision and deep learning. The tool detects and identifies surgical instruments in real-time, providing valuable insights and enhancing the efficiency and safety of surgical procedures.

## Features
- **Real-Time Object Detection:** Utilizes YOLOv8 for accurate and fast detection of surgical instruments.
- **User-Friendly Interface:** Simple web interface for uploading images and viewing detection results.
- **History Tracking:** Maintains a database of previous detections for review and analysis.
- **Custom Model Training:** Includes scripts and configuration for training custom models on new datasets.

## Project Structure
```
├── app.py                  # Main application file (Flask app)
├── requirements.txt        # Python dependencies
├── best.pt                 # Trained YOLOv8 model weights
├── instance/
│   └── history.db          # SQLite database for detection history
├── model_training/
│   ├── camera.py           # Camera capture utility
│   ├── data.yaml           # Dataset configuration
│   ├── image_detector_app.py # Training and detection script
│   ├── object_d.py         # Object detection logic
│   ├── requirements.txt    # Training dependencies
│   └── yolov8s.pt          # YOLOv8 base model weights
├── static/
│   └── uploads/            # Uploaded images
├── templates/
│   ├── index.html          # Main UI
│   ├── results.html        # Detection results
│   ├── history.html        # Detection history
│   ├── login.html          # User authentication
│   ├── layout.html         # Base template
│   └── gemini_results.html # Additional results page
```

## Getting Started
### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/italhachaudhary/sat.git
   cd sat
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. (Optional) Install training dependencies:
   ```sh
   pip install -r model_training/requirements.txt
   ```

### Running the Application
```sh
python app.py
```
The app will be available at `http://127.0.0.1:5000/`.

## Usage
- Upload an image via the web interface.
- View detected instruments and results.
- Access detection history for previous uploads.

## Model Training
- Place your dataset in the `model_training/` directory.
- Update `data.yaml` with your dataset paths.
- Use `image_detector_app.py` to train or test models.

## License
This project is for academic and research purposes. Please contact the author for commercial use.

## Acknowledgements
- YOLOv8 by Ultralytics
- Flask Web Framework
- Open Source Community
