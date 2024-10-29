import tensorflow as tf

# Load the original model from JSON and H5 weights
with open('final_model.json', 'r') as json_file:
    model_json = json_file.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights('final_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# No optimizations so that models should change output shapes.
converter.optimizations = []

# Convert the model to TensorFlow Lite
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('final_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Print model input and output details for verification
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"TFLite model input shape: {input_details[0]['shape']}")
print("TFLite model output shapes:")
for i, output_detail in enumerate(output_details):
    print(f"Output {i + 1}: {output_detail['shape']}")
