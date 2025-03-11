#!/bin/bash

echo "🔄 Merging model parts..."
python merge_model.py

echo "🚀 Starting Flask app..."
gunicorn -w 4 -b 0.0.0.0:5000 app:app  # Adjust if needed
