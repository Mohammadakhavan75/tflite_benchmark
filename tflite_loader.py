import tensorflow as tf
import numpy as np

class model_loader():
    def __init__(self, filepath):
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=filepath, experimental_delegates=[armnn_delegate])
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Test model on random input data.
        self.input_shape = self.input_details[0]['shape']
        

    def preprocessing(self, img):
        return tf.image.resize_images(img, (self.input_shape, self.input_shape))


    def inference(self, img):
        # BGR image to tensor
        input_tensor = self._pre_processing(img)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        # set tensor and invoke
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output_data
