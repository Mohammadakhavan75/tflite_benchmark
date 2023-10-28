import cv2
import numpy as np
import tensorflow as tf
# TODO: I have to make shape of image automatic
class data_loader2():
    def __init__(self, img_shape, model_shape, data_queue):
        self.img_shape = img_shape
        self.model_shape = model_shape
        self.data_queue = data_queue
        self.ret = True
        

    def load_img(self, path):
        img_np = cv2.imread(path)
        img_np = cv2.resize(img_np, (self.img_shape, self.img_shape))
        print(f"{img_np.shape[0]}, {self.img_shape}, {img_np.shape[0]!=self.img_shape}")
        if img_np.shape[0] != self.model_shape[1]:
            img_np = img_np.transpose(2, 0, 1)
        self.img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)
        self.img_tf = tf.expand_dims(self.img_tf , axis=0)
        return [self.img_tf]
    

    def load_vid(self, path):
        self.img_list = []
        # Open the video file
        video_capture = cv2.VideoCapture(path)  # Replace 'your_video.mp4' with your video file's path
        
        # Check if the video file was successfully opened
        if not video_capture.isOpened():
            print("Error: Could not open the video file.")
            exit()

        # Loop through the video frames
        while True:
            self.ret, frame = video_capture.read()  # Read a frame from the video

            # Check if the frame was successfully read
            if not self.ret:
               break  # Break the loop if the video has ended

            # Process and display the frame (e.g., you can show it using cv2.imshow)
            img_np = cv2.resize(frame, (self.img_shape, self.img_shape))
            if img_np.shape[0] != self.model_shape[1]:
                img_np = img_np.transpose(2, 0, 1)
            self.img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)
            self.img_tf = tf.expand_dims(self.img_tf , axis=0)
            self.data_queue.put(self.img_tf)
            # self.img_list.append(self.img_tf)

            # Exit the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close any open windows
        video_capture.release()
        return self.img_list
