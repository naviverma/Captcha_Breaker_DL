#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import argparse
import time

def decode_predictions(captcha_symbols, predictions):
    """
    Decode the model's predictions into a CAPTCHA string.
    Processes each position's predictions in order.
    """
    captcha_text = ''
    for pos_prediction in predictions:
        # Get the highest probability symbol for this position
        symbol_idx = np.argmax(pos_prediction[0])
        # Only add the symbol if it's not the padding character
        symbol = captcha_symbols[symbol_idx]
        if symbol != '£':  # Assuming '£' is the padding character
            captcha_text += symbol
    return captcha_text

def preprocess_image(image_path, target_width=192, target_height=96):
    """
    Preprocess image consistently with training.
    """
    # Read and convert image
    raw_data = cv2.imread(image_path)
    rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
    
    # Resize image
    processed_data = cv2.resize(rgb_data, (target_width, target_height))
    
    # Normalize
    processed_data = np.array(processed_data) / 255.0
    
    # Add batch dimension
    processed_data = np.expand_dims(processed_data, axis=0)
    
    return processed_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Directory containing CAPTCHA images to classify', type=str)
    parser.add_argument('--output', help='Output file to save classifications', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in CAPTCHAs', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the model to use.")
        exit(1)
    if args.captcha_dir is None:
        print("Please specify the directory with CAPTCHA images.")
        exit(1)
    if args.output is None:
        print("Please specify the output file for classifications.")
        exit(1)
    if args.symbols is None:
        print("Please specify the file with CAPTCHA symbols.")
        exit(1)

    # Enable MPS for GPU if available
    if tf.config.experimental.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
            print("MPS is enabled for GPU inference.")
        except RuntimeError as e:
            print(f"Error enabling MPS: {e}")
    else:
        print("No GPU detected. Using CPU for inference.")

    # Load symbols file
    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip() + '£'  # Adding the padding character to symbols

    # Load and compile model
    print(f"Loading model {args.model_name}...")
    with open(args.model_name + '.json', 'r') as json_file:
        model = keras.models.model_from_json(json_file.read())
    model.load_weights(args.model_name + '.h5')
    
    # Compile with the same settings as training
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
        metrics=['accuracy']
    )
    
    # Start timing the classification process
    start_time = time.time()

    with open(args.output, 'w') as output_file:
        for image_file in sorted(os.listdir(args.captcha_dir)):  # Sort to process in a consistent order
            if not image_file.endswith('.png'):
                continue

            # Get actual label from filename (assuming format "LABEL_*.png")
            actual_label = image_file.split('_')[0]
            
            # Preprocess image
            image_path = os.path.join(args.captcha_dir, image_file)
            processed_image = preprocess_image(image_path)
            
            # Get predictions
            predictions = model.predict(processed_image)
            
            # Decode predictions
            predicted_text = decode_predictions(captcha_symbols, predictions)
            
            # Write result
            output_file.write(f"{image_file}, {predicted_text}\n")
                
            # Print progress and current prediction
            print(f"Classified {image_file}: {predicted_text}") 

    # End timing the classification process
    end_time = time.time()
    total_time = end_time - start_time

    # Print the time taken for the entire classification process
    print(f"Total time taken for classification: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
