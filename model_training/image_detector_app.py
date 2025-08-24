import sys
import types
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

# Fix for YOLO + Torch in Streamlit
sys.modules["torch._classes"] = types.SimpleNamespace(__path__=[])

# Title
st.title("🔬 Surgical Assistance Tool")

# Choose input method
input_method = st.radio("Choose image source:", ("Upload", "Use Camera"))

uploaded_file = None

if input_method == "Upload":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
elif input_method == "Use Camera":
    uploaded_file = st.camera_input("Take a picture")
# Check if an image is uploaded
if uploaded_file is None:
    st.warning("Please upload an image or take a picture using the camera.")
    st.stop()
    


# Load YOLO model (adjust path as needed)
@st.cache_resource
def load_model():
    return YOLO(r"C:\Users\chtal\Desktop\SAT\best.pt")

model = load_model()

# Custom labels for class IDs
custom_labels = {
    0: "Scalpel Handle",
    1: "Tweezers",
    2: "Straight Scissors",
    3: "Curved Scissors",
}

# Detailed descriptions of each tool
tool_usages = {
    "Scalpel Handle": """The scalpel handle is a reusable surgical instrument designed to hold various types of scalpel blades. It provides a secure grip for the surgeon, allowing precise control during incisions. The handle comes in different sizes (commonly #3, #4, etc.) and is used to accommodate different blade shapes depending on the procedure. It is essential in almost all types of surgeries including general, orthopedic, cardiovascular, and plastic surgery.""",

    "Tweezers": """Surgical tweezers, also known as forceps, are fine precision tools used to grasp, hold, or manipulate tissues and other small objects during surgery. They are vital for tasks requiring delicate control, such as suturing, removing debris, or isolating tissue structures. There are various types including toothed, non-toothed, and dressing forceps, each suited to specific surgical scenarios.""",

    "Straight Scissors": """Straight surgical scissors are primarily used for cutting body tissues near the surface of a wound or for cutting sutures and materials. They offer direct, clean cuts and are typically used in procedures where precision is required in linear or superficial incisions. Mayo and Metzenbaum scissors are two common types, with Mayo being more robust and Metzenbaum used for more delicate dissection.""",

    "Curved Scissors": """Curved surgical scissors are designed for cutting deeper tissues or for navigating around anatomical curves. The curvature allows the surgeon to maintain visibility and access within confined or complex anatomical spaces. They're particularly useful in dissection, allowing for precise removal of tissue while minimizing damage to surrounding structures. Common types include curved Mayo and curved Metzenbaum scissors, each optimized for specific surgical tasks.""",
}

# Main logic
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Convert to NumPy (OpenCV) format
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Perform prediction
    results = model.predict(source=image_bgr, save=False, conf=0.5)

    detected_tools = set()

    # Annotate image
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label_name = custom_labels.get(class_id, "Unknown")
        label = f"{label_name} ({confidence:.2f})"

        # Draw bounding box and label
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Track detected tool names
        detected_tools.add(label_name)

    # Convert back to RGB for display
    result_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption="Detected Instruments")

    # Show usage information
    if detected_tools:
        st.markdown("### 🛠 Tool Usage Information")
        for tool in detected_tools:
            usage = tool_usages.get(tool, "No usage information available.")
            with st.expander(f"ℹ️ {tool}"):
                st.markdown(usage)

    # Save and offer result for download
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    result_pil = Image.fromarray(result_image)
    result_pil.save(temp_file.name)

    with open(temp_file.name, "rb") as file:
        st.download_button(
            label="📥 Download Result Image",
            data=file,
            file_name="detection_result.jpg",
            mime="image/jpeg"
        )
