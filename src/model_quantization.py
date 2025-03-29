import tensorflow as tf
import os

def convert_to_tflite(model, model_path, quantized=False):
    """Converts a Keras model to TensorFlow Lite format to reduce model size and improve inference speed.
    
    This function supports both standard conversion and quantized conversion for further compression.
    
    Args:
        model (tf.keras.Model): The trained Keras model.
        model_path (str): Path to save the converted model.
        quantized (bool): Whether to apply post-training quantization. Defaults to False.
    
    Returns:
        None: Saves the converted model to the specified path.
    """
    
    assert isinstance(model, tf.keras.Model), "model must be a TensorFlow Keras model"
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT] # Enables weight quantization
    
    tflite_model = converter.convert()
    
    file_name = "quantized_model.tflite" if quantized else "model.tflite"
    file_path = os.path.join(model_path, file_name)
    with open(file_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model saved at: {file_path}")


def load_tflite_model(model_path):
    """Loads a TFLite model and prepares it for inference."""
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model Input Shape:", input_details[0]['shape'])
    print("Model Output Shape:", output_details[0]['shape'])
    return interpreter
