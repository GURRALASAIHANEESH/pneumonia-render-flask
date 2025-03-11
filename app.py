import os
import numpy as np
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from skimage import filters, morphology

# Set up Flask app
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Log the current working directory
print(f"📂 Current working directory: {os.getcwd()}")

MODEL_PATH = "model.tflite"

# ✅ Step 1: Merge model parts if necessary
def merge_model_parts():
    """Merges split model files into a single model.tflite"""
    model_parts = sorted([f for f in os.listdir() if f.startswith("model.tflite.part")])
    
    if not model_parts:
        print("❌ No model parts found! Ensure they are uploaded.")
        return False

    print("🔄 Merging model parts...")
    with open(MODEL_PATH, "wb") as full_model:
        for part in model_parts:
            with open(part, "rb") as f:
                full_model.write(f.read())

    print("✅ Model successfully merged into model.tflite")
    return True

# Merge model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    merge_success = merge_model_parts()
    if not merge_success:
        print("⚠️ Model merging failed. Ensure all model parts are present!")

# ✅ Step 2: Load the TFLite model
if os.path.exists(MODEL_PATH):
    print("✅ Model found, loading...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded successfully!")
else:
    interpreter = None
    print("❌ Model not found! Ensure 'model.tflite' is present or check the merging process.")

# ✅ Step 3: Image Processing & Prediction
def get_className(classNo):
    return "Normal" if classNo == 0 else "Pneumonia"

def segment_pneumonia(image):
    """Applies Otsu thresholding and morphological operations to extract pneumonia-affected regions"""
    thresh_val = filters.threshold_otsu(image)
    binary_mask = image > thresh_val
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=30)
    return cleaned_mask

def calculate_pneumonia_percentage(image):
    """Calculates the percentage of lung affected by pneumonia"""
    pneumonia_mask = segment_pneumonia(image)
    total_lung_area = np.sum(image > 0)
    pneumonia_area = np.sum(pneumonia_mask)
    return (pneumonia_area / total_lung_area) * 100 if total_lung_area > 0 else 0

def get_result(file_path):
    """Runs the TFLite model on an input image and returns the classification result"""
    try:
        if interpreter is None:
            return "❌ Error: Model not loaded."

        # Load and preprocess image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(image, (128, 128))
        input_img = np.expand_dims(cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB), axis=0).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_img)
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]['index'])
        class_name = get_className(np.argmax(result, axis=1))

        # Compute pneumonia percentage if detected
        pneumonia_percentage = None
        if class_name == "Pneumonia":
            pneumonia_percentage = calculate_pneumonia_percentage(resized_img)

        return (f"{class_name} detected. Pneumonia affects {pneumonia_percentage:.2f}% of the lung."
                if pneumonia_percentage is not None else f"{class_name} detected.")
    except Exception as e:
        return f"❌ An error occurred: {str(e)}"
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# ✅ Step 4: Flask Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    f = request.files.get('file')
    if f and f.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)
        return get_result(file_path)
    return '❌ File type not allowed'

# ✅ Step 5: Start Flask Server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Ensure correct port binding
    app.run(debug=True, host='0.0.0.0', port=port)
