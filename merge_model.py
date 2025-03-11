import os

# Define the parts and output file
output_file = "model.tflite"
part_files = [f"model_part_{i:02d}" for i in range(11)]  # Adjust if the number of parts is different

# Check if the model already exists to avoid unnecessary merging
if not os.path.exists(output_file):
    with open(output_file, "wb") as outfile:
        for part in part_files:
            with open(part, "rb") as infile:
                outfile.write(infile.read())

    print("✅ Model reassembled successfully as 'model.tflite'!")
else:
    print("✅ Model already exists. No need to merge.")
