#!/bin/bash

# Step 1: Run retrieve_file_list.py
echo "Running retrieve_file_list.py..."
python3 retrieve_file_list.py

# Step 2: Run download_script.py
echo "Running download_script.py..."
python3 download_script.py

# Step 3: Run check_images_sizes.py
echo "Running check_images_sizes.py..."
python3 check_images_sizes.py

# Step 4: Generate training data
echo "Generating training data..."
python3 generate.py --width 192 --height 96 --count 100000 --output-dir train_data --symbols symbols.txt

# Step 5: Generate validation data
echo "Generating validation data..."
python3 generate.py --width 192 --height 96 --count 30000 --output-dir validation_data --symbols symbols.txt

# Step 6: Train the model
echo "Training the model..."
python3 train.py --width 192 --height 96 --length 6 --batch-size 32 --train-dataset ./train_data --validate-dataset ./validation_data --output-model-name final_model --epochs 15 --symbols symbols.txt

# Step 7: Convert H5 model to TFLite
echo "Converting final_model.h5 to TFLite format..."
python3 convert_h5_to_tflite.py

# Step 8: Run classify.py
echo "Classifying CAPTCHA images..."
python3 classify.py --model-name final_model --captcha-dir captchas --output output_last.txt --symbols symbols.txt

# Step 9: Convert output to CSV
echo "Converting output to CSV format..."
python3 convert_to_csv.py nasingh output_last.txt last_submission.csv

# Step 10: Open output_final.txt in Vim
echo "Opening output_final.txt in Vim..."
vim output_final.txt

echo "All steps completed successfully!"
