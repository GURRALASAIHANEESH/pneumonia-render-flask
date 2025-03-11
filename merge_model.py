with open("model.tflite", "wb") as output_file:
    for i in range(11):  # Since you split into 11 parts
        part_filename = f"model_part_{i:02}"
        with open(part_filename, "rb") as part_file:
            output_file.write(part_file.read())

print("model.tflite successfully reassembled!")
