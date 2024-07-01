import gradio as gr
import cv2
import tensorflow as tf
import warnings
import numpy as np  

def process_video(video):
  # Your video processing logic here
  # ...
    # C:\Users\aumal\AppData\Local\Temp\gradio\226aed1955c3b5d15dc21bcd80cbfe5740309643\sample.webm
    # print(type(video),video)
    # return video
    cap = cv2.VideoCapture(video)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        # frame = cv2.imread(frame)#tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    frames = np.array(frames)
    cap.release()
    size = 46,140
    duration = 3
    fps = 25
    out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for _ in range(fps * duration):
        data = frames[_]
        print(_)
        out.write(data)
    out.release()
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    frames = tf.cast((frames - mean), tf.float32) / std

    # frames = tf.expand_dims(frames,axis = 0)
    print('frames shape' ,frames.shape)
    print(video,out)
    return video,"bbaf2n.mpg"

interface = gr.Interface(
  fn=process_video,
  inputs=gr.Video(),
  outputs=[gr.Video(),gr.Text()],
  title="Video Processing",
  description="Upload a video or use your webcam for processing."
)

interface.launch()
