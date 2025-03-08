from tensorflow.keras.models import load_model
import tensorflow as tf

# Load your existing model
model = load_model("vgg_unfrozen.h5")

# Convert model to TFLite for smaller size
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save new lightweight model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model successfully converted to TFLite format!")
