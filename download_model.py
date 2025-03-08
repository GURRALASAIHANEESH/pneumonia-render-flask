import os
import gdown

model_path = "vgg_unfrozen.h5"

# Check if model already exists
if not os.path.exists(model_path):
    file_url = "https://drive.google.com/uc?id=1EoyanPvpR2BkkH3LKTSGDm-hRboGYeQO"
    gdown.download(file_url, model_path, quiet=False)
    print("✅ Model downloaded successfully!")
else:
    print("✅ Model already exists, skipping download.")
