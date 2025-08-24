import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import types
import sys
import google.generativeai as genai
import json
import base64

# Fix for a known issue with YOLO and PyTorch in a multithreaded Flask environment
sys.modules["torch._classes"] = types.SimpleNamespace(__path__=[])

app = Flask(__name__)
app.secret_key = 'your_very_secret_key'  # IMPORTANT: Change this for production
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# --- Gemini API Configuration ---
# IMPORTANT: Replace "YOUR_GEMINI_API_KEY_HERE" with your actual Google AI Studio API key.
GEMINI_API_KEY = "" # Replace with your key
try:
    genai.configure(api_key=GEMINI_API_KEY)
    
    if "YOUR_GEMINI_API_KEY_HERE" in GEMINI_API_KEY:
         print("WARNING: Gemini API key is not set. Please replace 'YOUR_GEMINI_API_KEY_HERE' in app.py with your actual key.")
    else:
        print("Gemini API configured successfully.")

except Exception as e:
    print(f"FATAL: Error configuring Gemini API. Make sure the API key is correct and you have network access.")
    print(f"Error details: {e}")


# Create the upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
oauth = OAuth(app)

# --- Database Model (Updated) ---
# NOTE: You may need to delete your old 'history.db' file for these changes to apply
class History(db.Model):
    """Represents an entry in the analysis history for both YOLO and Gemini."""
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    image_filename = db.Column(db.String(100), nullable=False)
    result_image_filename = db.Column(db.String(100), nullable=True) # Specific to YOLO
    analysis_type = db.Column(db.String(50), nullable=False) # 'YOLO' or 'Gemini'
    analysis_result = db.Column(db.String(2000), nullable=False) # YOLO tools or Gemini JSON

    def __repr__(self):
        return f'<History {self.id} {self.user_email}>'

# --- Google OAuth Configuration ---
google = oauth.register(
    name='google',
    client_id='', # Your Google OAuth client ID
    client_secret='', # Your Google OAuth client secret
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'openid email profile'},
    jwks_uri="https://www.googleapis.com/oauth2/v3/certs"
)

# --- Model Loading and Configuration ---
def load_yolo_model():
    """Loads the YOLOv8 model from the specified path."""
    try:
        print("Loading YOLO model...")
        # Assuming 'best.pt' is in the same directory as app.py or accessible
        model = YOLO('best.pt')
        print("YOLO model loaded successfully.")
        return model
    except Exception as e:
        print(f"FATAL: Error loading YOLO model: {e}")
        return None

model = load_yolo_model()

custom_labels = {0: "Scalpel Handle", 1: "Tweezers", 2: "Straight Scissors", 3: "Curved Scissors"}
tool_usages = {
    "Scalpel Handle": "Serves as the primary interface for holding scalpel blades, designed for precision and ergonomic comfort. The handle's weighted design provides balance and control, critical for making accurate cuts through different layers of tissue. It allows for the secure attachment of various disposable blades, each tailored for specific incision types.",
    "Tweezers": "Used for grasping, holding, and manipulating delicate tissues with minimal trauma. Different types, such as Adson (with teeth) for skin or DeBakey (atraumatic) for vessels, are chosen based on the tissue's sensitivity. They are essential for stabilizing tissue during dissection, suturing, or for the delicate handling of needles and other small items.",
    "Straight Scissors": "Ideal for making clean, linear cuts through sutures, dressings, and tough or superficial tissues on a flat plane. Their design provides maximum mechanical advantage for powerful, straight-line incisions. They are commonly used for tasks that require precision cutting without the need to navigate around anatomical curves, such as preparing grafts or trimming wound edges.",
    "Curved Scissors": "Designed to navigate complex anatomical contours, allowing surgeons to cut around organs and vessels with enhanced visibility and safety. The curvature of the blades follows the natural lines of the body, which is especially useful in deep or confined surgical fields. This design minimizes the risk of inadvertently damaging adjacent structures while dissecting or transecting tissue."
}

# --- Image Processing Helper ---
def resize_image(image_path, max_size=(1280, 1280)):
    """Resizes an image to a maximum size while maintaining aspect ratio."""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            img.save(image_path)
        return True
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return False

# --- Gemini API Function ---
def analyze_image_with_gemini(image_path):
    """Analyzes an image using the Gemini Pro Vision model."""
    try:
        print(f"Gemini Analysis: Starting analysis for {image_path}")
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        with Image.open(image_path) as img:
            prompt = "Identify the surgical instrument in this image. Provide its name and a detailed 3 to 4-line description of its primary medical usage. Format the output strictly as: 'Instrument Name: [Name]\\nUsage: [Detailed description]'"
            print("Gemini Analysis: Sending prompt and image to Gemini API...")
            response = gemini_model.generate_content([prompt, img])
            print("Gemini Analysis: Gemini API response received.")
            # Ensure the response is not empty or malformed
            if not response.text:
                raise ValueError("Gemini API returned an empty response.")
            print(f"Gemini Analysis: Raw response text: {response.text}")
            return response.text
    except Exception as e:
        print(f"Gemini Analysis: Error during Gemini API call or response processing: {e}")
        raise e


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login/google')
def login_google():
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    try:
        token = google.authorize_access_token()
        user_info = google.get('userinfo').json()
        session['user'] = user_info
        return redirect(url_for('main_app'))
    except Exception as e:
        print(f"Authorization error: {e}")
        flash('Authentication failed. Please try again.')
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/app')
def main_app():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('index.html', user=session.get('user'))

def handle_file_upload():
    """Helper function to handle file upload and validation."""
    if 'user' not in session:
        return None, redirect(url_for('index'))
    if 'file' not in request.files:
        flash('No file part in the request.')
        return None, redirect(url_for('main_app'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected.')
        return None, redirect(url_for('main_app'))
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath, None
    return None, redirect(url_for('main_app'))


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handles image upload for YOLO object detection."""
    filepath, error_response = handle_file_upload()
    if error_response:
        return error_response
    
    if not model:
        flash('The YOLO analysis model is not loaded. Please contact the administrator.')
        return redirect(url_for('main_app'))

    try:
        filename = os.path.basename(filepath)
        print("YOLO Analysis: Resizing image...")
        if not resize_image(filepath):
             flash('Could not process the image file. It might be corrupted.')
             return redirect(url_for('main_app'))

        print("YOLO Analysis: Performing YOLO detection...")
        results = model(filepath)
        image = cv2.imread(filepath)
        detected_tools = set()

        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label_name = custom_labels.get(class_id, "Unknown Tool")
                
                label_text = f"{label_name} ({confidence:.2f})"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                detected_tools.add(label_name)

        result_image_filename = 'result_' + filename
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_image_filename)
        cv2.imwrite(result_image_path, image)

        detection_details = [{"name": tool, "usage": tool_usages.get(tool, "No usage info.")} for tool in sorted(list(detected_tools))]

        # Save to history
        user_email = session['user'].get('email')
        if user_email:
            tools_str = ', '.join(sorted(list(detected_tools))) if detected_tools else "None"
            new_history = History(user_email=user_email, image_filename=filename, result_image_filename=result_image_filename, analysis_type='YOLO', analysis_result=tools_str)
            db.session.add(new_history)
            db.session.commit()
            print("YOLO Analysis: Saved YOLO analysis to history.")

        return render_template('results.html', user=session.get('user'), image_path=f'uploads/{result_image_filename}', results=detection_details)

    except Exception as e:
        print(f"YOLO Analysis: An error occurred during YOLO analysis: {e}")
        flash(f'An unexpected error occurred during YOLO analysis: {e}')
        return redirect(url_for('main_app'))


@app.route('/gemini_upload', methods=['POST'])
def gemini_upload_image():
    """Handles image upload for Gemini analysis."""
    print("Gemini Upload: Received request for Gemini analysis.")
    if "YOUR_GEMINI_API_KEY_HERE" in GEMINI_API_KEY:
        flash("CONFIGURATION ERROR: The Gemini API key is not set in 'app.py'. Please update it.")
        print("Gemini Upload: API key not set, redirecting.")
        return redirect(url_for('main_app'))
        
    filepath, error_response = handle_file_upload()
    if error_response:
        print(f"Gemini Upload: File upload error: {error_response}")
        return error_response

    try:
        filename = os.path.basename(filepath)
        print(f"Gemini Upload: File saved to {filepath}. Starting Gemini analysis...")
        analysis_result = analyze_image_with_gemini(filepath)
        print("Gemini Upload: Gemini analysis completed. Parsing result...")

        instrument_name = "Could not identify"
        usage = "No usage information available."

        # Attempt to parse the structured response
        if "Instrument Name:" in analysis_result and "Usage:" in analysis_result:
            try:
                name_part, usage_part = analysis_result.split("Usage:", 1)
                instrument_name = name_part.replace("Instrument Name:", "").strip()
                usage = usage_part.strip()
                print(f"Gemini Upload: Successfully parsed instrument: {instrument_name}, usage: {usage[:50]}...") # Log first 50 chars of usage
            except ValueError as ve:
                print(f"Gemini Upload: Error parsing Gemini response format: {ve}. Raw response: {analysis_result}")
                flash("Gemini analysis returned an unexpected format. Please try again.")
                return redirect(url_for('main_app'))
        else:
            print(f"Gemini Upload: Gemini response did not contain expected 'Instrument Name:' or 'Usage:'. Raw response: {analysis_result}")
            flash("Gemini analysis could not extract instrument name and usage. Raw response might be in an unexpected format.")
            # Fallback to showing raw result if parsing fails completely
            instrument_name = "Analysis Result (unparsed)"
            usage = analysis_result


        # Save to history
        user_email = session['user'].get('email')
        if user_email:
            analysis_data = {"instrument_name": instrument_name, "usage": usage}
            new_history = History(user_email=user_email, image_filename=filename, result_image_filename=filename, analysis_type='Gemini', analysis_result=json.dumps(analysis_data))
            db.session.add(new_history)
            db.session.commit()
            print("Gemini Upload: Saved Gemini analysis to history.")

        return render_template('gemini_results.html', user=session.get('user'), image_path=f'uploads/{filename}', instrument_name=instrument_name, usage=usage)

    except Exception as e:
        print(f"Gemini Upload: An error occurred during Gemini analysis: {e}")
        flash(f'An unexpected error occurred during Gemini analysis: {e}')
        return redirect(url_for('main_app'))

@app.route('/live_camera_feed', methods=['POST'])
def live_camera_feed():
    """
    Receives base64 image data from the frontend, performs YOLO detection,
    and returns the annotated image and detected tools as JSON.
    """
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if not model:
        return jsonify({"error": "YOLO model not loaded."}), 500

    try:
        data = request.get_json()
        image_data = data['image_data']

        # Remove the "data:image/jpeg;base64," prefix
        base64_img = image_data.split(',')[1]
        
        # Decode base64 string to bytes
        img_bytes = base64.b64decode(base64_img)
        
        # Convert bytes to numpy array, then to OpenCV image
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not decode image."}), 400

        # Perform inference
        results = model.predict(source=image, save=False, conf=0.6) # Lower confidence for live feed

        detected_tools_info = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                label_name = custom_labels.get(class_id, "Unknown Tool")
                label_text = f"{label_name} ({confidence:.2f})"
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label background
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 255, 0), -1)
                # Draw label text
                cv2.putText(image, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                detected_tools_info.append({
                    "name": label_name,
                    "usage": tool_usages.get(label_name, "No usage information available.")
                })
        
        # Encode the processed image back to base64
        _, buffer = cv2.imencode('.jpg', image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "processed_image_data": f"data:image/jpeg;base64,{processed_image_base64}",
            "detected_tools": detected_tools_info
        })

    except Exception as e:
        print(f"Error during live camera feed processing: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500


@app.route('/history')
def history():
    """Displays the user's analysis history."""
    if 'user' not in session:
        return redirect(url_for('index'))

    user_email = session['user'].get('email')
    processed_history = []
    if user_email:
        history_items = History.query.filter_by(user_email=user_email).order_by(History.timestamp.desc()).all()
        for item in history_items:
            if item.analysis_type == 'Gemini':
                try:
                    # Add parsed JSON to the item object for easy access in the template
                    item.parsed_result = json.loads(item.analysis_result)
                except json.JSONDecodeError:
                    item.parsed_result = {"instrument_name": "Error", "usage": "Could not parse result."}
            processed_history.append(item)

    return render_template('history.html', user=session.get('user'), history=processed_history)


if __name__ == '__main__':
    with app.app_context():
        print("Creating database tables if they don't exist...")
        db.create_all()
        print("Database tables created.")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

