import tensorflow as tf
# import tflite_runtime.interpreter as tflite

class model_loader():
    def __init__(self, args):
        # Load TFLite model and allocate tensors.
        if args.armnn_delegate is None:
            self.interpreter = tf.lite.Interpreter(model_path=args.model_path)
        else:
            self.interpreter = tf.lite.Interpreter(model_path=args.model_path, experimental_delegates=[args.armnn_delegate])
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Test model on random input data.
        self.input_shape = self.input_details[0]['shape'][2]
        self.model_shape = self.input_details[0]['shape']
        
    
    def inference(self, input_tensor):
        # BGR image to tensor
        if input_tensor.shape != self.model_shape:
            raise ValueError(f"Input data shape {input_tensor.shape} does not match the expected shape {self.model_shape}")

        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        # set tensor and invoke
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output_data
