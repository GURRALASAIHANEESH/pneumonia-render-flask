import os
import numpy as np
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from skimage import filters, morphology
from tensorflow.keras.models import load_model

# Set up Flask app
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Log the current working directory to check where the app is running from
print(f"Current working directory: {os.getcwd()}")

# Convert Keras model to TFLite format
def convert_model_to_tflite():
    # Load your existing model
    model = load_model("vgg_unfrozen.h5")

    # Convert model to TFLite for smaller size
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save new lightweight model
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    print("✅ Model successfully converted to TFLite format!")

# Load TFLite model
MODEL_PATH = "model.tflite"
print(f"Checking model file at: {MODEL_PATH}")  # Log model path

if os.path.exists(MODEL_PATH):
    print("✅ Model found.")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('✅ Model loaded successfully!')
else:
    interpreter = None
    print('⚠️ Model not found! Ensure "model.tflite" is in the directory.')

def get_className(classNo):
    return "Normal" if classNo == 0 else "Pneumonia"

def segment_pneumonia(image):
    thresh_val = filters.threshold_otsu(image)
    binary_mask = image > thresh_val
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=30)
    return cleaned_mask

def calculate_pneumonia_percentage(image):
    pneumonia_mask = segment_pneumonia(image)
    total_lung_area = np.sum(image > 0)
    pneumonia_area = np.sum(pneumonia_mask)
    return (pneumonia_area / total_lung_area) * 100 if total_lung_area > 0 else 0

def get_result(file_path):
    try:
        if interpreter is None:
            return "Error: Model not loaded."

        # Load and preprocess image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(image, (128, 128))
        input_img = np.expand_dims(cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB), axis=0)
        input_img = np.array(input_img, dtype=np.float32)  # Ensure correct format

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
        return f"An error occurred: {str(e)}"
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

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
    return 'File type not allowed'

if __name__ == '__main__':
    # Convert the model to TFLite format if not already done
    if not os.path.exists(MODEL_PATH):
        convert_model_to_tflite()

    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))  # Ensure correct port binding
    app.run(debug=True, host='0.0.0.0', port=port)
