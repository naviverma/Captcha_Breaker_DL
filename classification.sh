#!/bin/bash

# Step 1: Run retrieve_file_list.py
echo "Running retrieve_file_list.py..."
python3 retrieve_file_list.py
echo "The file is successfully retrieved and stored as file_list.txt."

# Step 2: Run download_script.py
echo "Running download_script.py..."
python3 download_script.py
echo "The CAPTCHAs are successfully saved in the captchas folder."

# Step 3: Run check_images_sizes.py
echo "Running check_images_sizes.py..."
python3 check_images_sizes.py

# Step 4: Convert H5 model to TFLite
echo "Converting final_model.h5 to TFLite format..."
python3 convert_h5_to_tflite.py
echo "The Model is converted to tflite for raspberry pi compatibility"

# Step 5: Run classify.py
echo "Classifying CAPTCHA images..."
python3 classify.py --model-name final_model --captcha-dir captchas --output output_last.txt --symbols symbols.txt

# Step 6: Convert output to CSV
echo "Converting output to CSV format..."
python3 convert_to_csv.py nasingh output_last.txt last_submission.csv

# Step 7: Open output_final.txt in Vim
echo "Opening output_final.txt in Vim..."
vim output_final.txt

echo "All steps completed successfully!"