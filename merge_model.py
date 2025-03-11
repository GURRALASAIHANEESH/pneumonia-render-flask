import os

def merge_model_parts():
    model_parts = sorted([f for f in os.listdir() if f.startswith("model.tflite.part")])
    
    if not model_parts:
        print("❌ No model parts found!")
        return False

    with open("model.tflite", "wb") as full_model:
        for part in model_parts:
            with open(part, "rb") as f:
                full_model.write(f.read())

    print("✅ Model parts successfully merged into model.tflite")
    return True

if __name__ == "__main__":
    success = merge_model_parts()
    if not success:
        print("⚠️ Model merging failed. Ensure model parts are uploaded.")
