#!/usr/bin/env python3

import os
import random
import argparse
from captcha.image import ImageCaptcha

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int, required=True)
    parser.add_argument('--height', help='Height of captcha image', type=int, required=True)
    parser.add_argument('--count', help='How many captchas to generate', type=int, required=True)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str, required=True)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str, required=True)
    args = parser.parse_args()

    # Load the fonts for the CAPTCHA generator
    fonts = ['/The Jjester.otf', '/eamonwide.woff.ttf']  # Paths to fonts
    captcha_generator = ImageCaptcha(width=args.width, height=args.height, fonts=fonts)

    # Augmentation settings
    captcha_generator.character_warp_dx = (0.1, 0.5)
    captcha_generator.character_warp_dy = (0.2, 0.5)
    captcha_generator.character_rotate = (-45, 45)
    captcha_generator.word_space_probability = 0.5

    # Read the symbols from the provided symbols file
    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    print("Generating captchas with symbol set: " + captcha_symbols)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate the specified number of captchas
    for i in range(args.count):
        print(f"Generating image no {i+1}")

        # Generate a random string of length between 1 and 6
        captcha_length = random.randint(1, 6)
        random_str = ''.join(random.choices(captcha_symbols, k=captcha_length))

        # Generate the file names
        image_id = f"captcha_{i+1}"
        image_path = os.path.join(args.output_dir, image_id + '.png')
        text_path = os.path.join(args.output_dir, image_id + '.txt')

        # Generate the CAPTCHA image with augmentation
        captcha_image = captcha_generator.generate_image(random_str)
        captcha_image.save(image_path)

        # Save the CAPTCHA text to a .txt file
        with open(text_path, 'w') as text_file:
            text_file.write(random_str)

    print("Finished generating captchas.")

if __name__ == '__main__':
    main()
