import os

def merge_model_parts():
    model_parts = sorted([f for f in os.listdir() if f.startswith("model_part_")])
    
    if not model_parts:
        print("❌ No model parts found! Ensure they are uploaded.")
        return False

    with open("model.tflite", "wb") as full_model:
        for part in model_parts:
            with open(part, "rb") as f:
                full_model.write(f.read())

    print("✅ Model parts successfully merged into model.tflite")
    return True

if __name__ == "__main__":
    if merge_model_parts():
        print("✅ Model merged successfully.")
    else:
        print("⚠️ Model merging failed. Ensure all model parts are uploaded to the repository.")
