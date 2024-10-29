<<<<<<< HEAD
# Captcha_Breaker_DL
=======
# TEAM 44.

#### Team Members:  
1. Navdeep Singh 22331003

#### Team Size :1 
# Project_2 Scalable_Computing.
Captcha Indentification.  
## Prerequisites

To ensure compatibility, please use the following dependencies:

- **Python**: 3.8.12
- **TensorFlow**: 2.13.0
- **OpenCV**: 4.10.0.84
- **NumPy**: 1.24.3
- **Matplotlib**: 3.7.5
- **Captcha**: 0.6.0
- **Pillow**: 10.4.0

You can install these dependencies using the provided `requirements.txt` file.

## Instructions

### Running CAPTCHA Classification Only

To perform CAPTCHA classification, follow these steps:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Classification Script**:
   ```bash
   ./classification.sh
   ```
### Running Full Pipeline (Downloading, Generating training and validation data, Training, Classification)

To perform CAPTCHA classification, follow these steps:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Classification Script**:
   ```bash
   ./run.sh
   ```
### Manual Execution Steps for CAPTCHA Project

Follow these instructions to manually execute each part of the CAPTCHA project pipeline:

1. **Retrieve the File List:**  
   - Run the command: `python3 retrieve_file_list.py`  
   This will retrieve the list of files needed for the project and save them in `file_list.txt`.

2. **Download the Data:**  
   - Run the command: `python3 download_script.py`  
   This step will download the required files based on the retrieved file list.

3. **Check Image Sizes:**  
   - Run the command: `python3 check_images_sizes.py`  
   This will verify that all the downloaded images have consistent dimensions.

4. **Generate Training Data:**  
   - Run the command: `python3 generate.py --width 192 --height 96 --count 100000 --output-dir train_data --symbols symbols.txt`  
   This generates 100,000 CAPTCHA images with the specified width and height, using symbols listed in `symbols.txt`, and saves them in the `train_data` directory.

5. **Generate Validation Data:**  
   - Run the command: `python3 generate.py --width 192 --height 96 --count 30000 --output-dir validation_data --symbols symbols.txt`  
   This creates 30,000 CAPTCHA images for validation, with the same dimensions and symbols, saving them in the `validation_data` directory.

6. **Train the Model:**  
   - Run the command: `python3 train.py --width 192 --height 96 --length 6 --batch-size 32 --train-dataset ./train_data --validate-dataset ./validation_data --output-model-name final_model --epochs 15 --symbols symbols.txt`  
   This step trains a CAPTCHA recognition model using the generated training and validation datasets, setting the maximum CAPTCHA length to 6 characters, with a batch size of 32 for 15 epochs. The trained model will be saved as `final_model`.

7. **Convert the Trained Model to TFLite:**  
   - Run the command: `python3 convert_h5_to_tflite.py`  
   This converts the trained model (`final_model.h5`) to the TFLite format for deployment.

8. **Classify CAPTCHA Images:**  
   - Run the command: `python3 classify.py --model-name final_model --captcha-dir captchas --output output_last.txt --symbols symbols.txt`  
   This performs CAPTCHA classification on images located in the `captchas` directory using the trained model, saving the predictions in `output_last.txt`.

9. **Convert the Classification Output to CSV:**  
   - Run the command: `python3 convert_to_csv.py nasingh output_last.txt last_submission.csv`  
   This converts the classification results from `output_last.txt` to a CSV file named `last_submission.csv`.

10. **Open the Final Output in Vim:**  
    - Run the command: `vim output_final.txt`  
    This opens the `output_final.txt` file in Vim for review.

After completing all these steps, the pipeline for the CAPTCHA project will be executed successfully.
>>>>>>> 4280e04 (Captcha Breakeddd)
