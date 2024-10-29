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
import random
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers

# To predict variable length CAPTCHA (up to 6 characters)
def create_model(max_captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2, dropout_rate=0.2, l2_reg=0.001):
    input_tensor = keras.layers.Input(input_shape)
    x = input_tensor
    
    # Add convolutional layers with batch normalization, activation, and regularization
    for i, module_length in enumerate([module_size] * model_depth):
        for _ in range(module_length):
            x = keras.layers.Conv2D(
                64 * 2**min(i, 3), 
                kernel_size=3, 
                padding='same', 
                kernel_initializer='he_uniform',
                kernel_regularizer=regularizers.l2(l2_reg)  # L2 regularization
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)
        x = keras.layers.Dropout(dropout_rate)(x)  # Dropout layer
    
    x = keras.layers.Flatten()(x)

    # Create output layers for each character in the CAPTCHA
    output_layers = []
    for i in range(max_captcha_length):
        output = keras.layers.Dense(
            captcha_num_symbols, 
            activation='softmax', 
            name=f'char_{i+1}',
            kernel_regularizer=regularizers.l2(l2_reg)  # L2 regularization
        )(x)
        output_layers.append(output)
    
    # Create the model with multiple outputs
    model = keras.models.Model(inputs=input_tensor, outputs=output_layers)
    return model

# Custom callback to save the graph after each epoch
class SavePlotCallback(keras.callbacks.Callback):
    def __init__(self):
        super(SavePlotCallback, self).__init__()
        self.epoch = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        # Update the lists with current logs
        self.epoch.append(epoch)
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('char_1_accuracy'))
        self.val_acc.append(logs.get('val_char_1_accuracy'))

        # Create the plot and save it as an image
        plt.figure(figsize=(12, 6))

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(self.epoch, self.train_loss, label='Train Loss')
        plt.plot(self.epoch, self.val_loss, label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch, self.train_acc, label='Train Accuracy')
        plt.plot(self.epoch, self.val_acc, label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Save the plot as an image file after each epoch
        plt.tight_layout()
        plt.savefig(f'training_plot_epoch_{epoch + 1}.png')  # Save plot for each epoch
        plt.close()

# Data generator class for feeding images in batches
class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, max_captcha_length, captcha_symbols, captcha_width, captcha_height, padding_char='£'):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.max_captcha_length = max_captcha_length
        self.captcha_symbols = captcha_symbols + padding_char  # Add padding character to symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height
        self.padding_char = padding_char
        file_list = [f for f in os.listdir(self.directory_name) if f.endswith('.png')]  # Only process .png files
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.file_keys = list(self.files.keys())  # Preserve the original list of files
        self.count = len(file_list)

    def __len__(self):
        return int(np.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.captcha_height, self.captcha_width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, len(self.captcha_symbols)), dtype=np.uint8) for i in range(self.max_captcha_length)]

        for i in range(self.batch_size):
            # Check if the file list is empty and repopulate it
            if len(self.file_keys) == 0:
                self.file_keys = list(self.files.keys())

            random_image_label = random.choice(self.file_keys)
            self.file_keys.remove(random_image_label)  # Remove it from the copy, not from the original `self.files`
            random_image_file = self.files[random_image_label]

            # Preprocess the image
            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            processed_data = np.array(rgb_data) / 255.0  # Scale the input pixel values to [0,1]
            X[i] = processed_data

            # Read the corresponding text file for the label
            label_path = os.path.join(self.directory_name, random_image_label + '.txt')
            with open(label_path, 'r') as label_file:
                captcha_text = label_file.readline().strip()

            # Pad the label to the maximum length with the padding character if shorter
            captcha_text = captcha_text.ljust(self.max_captcha_length, self.padding_char)

            # One-hot encode the label
            for j, ch in enumerate(captcha_text):
                y[j][i, :] = 0
                y[j][i, self.captcha_symbols.find(ch)] = 1

        return X, y

def main():
    start_time = time.time()

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Max length of captchas in characters', type=int)
    parser.add_argument('--batch-size', help='Batch size for training', type=int)
    parser.add_argument('--train-dataset', help='Path to the training data', type=str)
    parser.add_argument('--validate-dataset', help='Path to the validation data', type=str)
    parser.add_argument('--output-model-name', help='Output model file name', type=str)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int)
    parser.add_argument('--symbols', help='Path to symbols file', type=str)
    parser.add_argument('--input-model', help='Pretrained model to continue training', type=str, default=None)
    args = parser.parse_args()

    # Load symbols
    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    # Create the model
    model = create_model(args.length, len(captcha_symbols) + 1, (args.height, args.width, 3))  # +1 for padding character

    # If provided, load the pretrained model
    if args.input_model is not None:
        model.load_weights(args.input_model)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(1e-4, amsgrad=True), metrics=['accuracy'])
    
    # Model summary
    model.summary()

    # Data generators
    training_data = ImageSequence(args.train_dataset, args.batch_size, args.length, captcha_symbols, args.width, args.height, padding_char='£')
    validation_data = ImageSequence(args.validate_dataset, args.batch_size, args.length, captcha_symbols, args.width, args.height, padding_char='£')

    # Initialize the plot saving callback
    save_plot_callback = SavePlotCallback()

    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint(args.output_model_name + '.h5', save_best_only=False),
        save_plot_callback  # Save the plot after each epoch
    ]

    # Save model architecture to JSON
    with open(args.output_model_name + ".json", "w") as json_file:
        json_file.write(model.to_json())

    try:
        model.fit(training_data,
                  validation_data=validation_data,
                  epochs=args.epochs,
                  callbacks=callbacks,
                  workers=1)
    except KeyboardInterrupt:
        print('Training interrupted, saving current weights as ' + args.output_model_name + '_resume.h5')
        model.save_weights(args.output_model_name + '_resume.h5')

    # Save training time
    end_time = time.time()
    time_taken = end_time - start_time
    with open('train_time_taken.txt', 'w') as file:
        file.write('Train: {:.2f} seconds'.format(time_taken))
        print('Time taken for training the model:', time_taken, 'seconds')

if __name__ == '__main__':
    main()
