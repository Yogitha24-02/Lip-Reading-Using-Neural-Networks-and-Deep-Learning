import gradio as gr
import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
from google.colab.patches import cv2_imshow
import os
import dlib


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


cropping_dictionary = {}
cropping_dictionary['s30'] = (200,246,80,220)
cropping_dictionary['s5'] = (210,256,120,260)
cropping_dictionary['s20'] = (200,246,120,260)
cropping_dictionary['s1'] = (190,236,80,220)

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss
tf.keras.utils.get_custom_objects()['CTCLoss'] = CTCLoss
model= load_model(r"Models\LipNet_on_grid.h5", custom_objects={'CTCLoss': CTCLoss})
print(model.summary())



def detect_box(lips):
    minX = 10**9
    minY = 10**9
    maxX = 0
    maxY = 0
    for x,y in lips:
      minX = min(minX,x)
      maxX = max(maxX,x)
      minY = min(minY,y)
      maxY = max(maxY,y)
    # print((minX,minY),(maxX,maxY))
    return minX,minY,maxX,maxY

def crop_image(minX,minY,maxX,maxY,image):
  height = maxY-minY
  width = maxX-minX
  # print(minX,minY,maxX,maxY)
  if height>46 or width>140:
    image = cv2.resize(image[minY:maxY,minX:maxX,:],(140,46))
    return image
  else:
    diffY = 46-height
    diffX = 140-width
    # print(diffY,diffX)
    # print(minY-diffY//2 - diffY%2 , maxY+diffY//2,minX - diffX//2 -diffX %2 , maxX + diffX//2)
    return image[minY-diffY//2 - diffY%2 : maxY+diffY//2,minX - diffX//2 -diffX %2 : maxX + diffX//2,:]
def detect_lips(image_path):
    predictor_path = '/content/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    gray = image_path
    print('shape',gray.shape)
    # try:
    print('Enter')
    try:
      faces = detector(gray)
      print('faces',faces)
      for face in faces:
        landmarks = predictor(gray,face)
        lips = [(landmarks.part(i).x ,landmarks.part(i).y) for i in range(48,68)]

      minX,minY,maxX,maxY = detect_box(lips)

      height = maxY-minY
      width = maxX-minX
      if height>46 or width>140:
        return
      diffY = 46-height
      diffX = 140-width
      minY = minY-diffY//2 - diffY%2
      maxY = maxY+diffY//2
      minX = minX - diffX//2 -diffX %2
      maxX = maxX + diffX//2
      print('Co-ordinates',minX,minY,maxX,maxY)
      cropped_image = (minX,minY,maxX,maxY)
    except:
      print('Unable to detect face')
    return cropped_image
    # except Exception as e:
    #   print('error')
    #   return 0,0,0,0



def predict(video):
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    cv2_imshow(frame)
    b,d,a,c = detect_lips(frame)
    print(a,b,c,d)
    a,b,c,d = cropping_dictionary['s1']
    print(a,b,c,d)
    if a==b==c==d==0:
      return 'Try with other video'
    print(d,c,b,a)
    frames = []
    cap = cv2.VideoCapture(video)
    for _ in range(min(75,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[d:c,b:a,:])
    cv2_imshow(np.array(frame[d:c,b:a,:]))
    # print(np.array(frames).shape)
    cap.release()
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    frames = tf.cast((frames - mean), tf.float32) / std
    frames = tf.expand_dims(frames,axis = 0)
    yhat = model1.predict(frames)
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    output= [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
    print('output',output)
    return output[0].numpy()




interface = gr.Interface(
  fn=predict,
  inputs=gr.Video(),
  outputs=gr.Text(),
  title="Video Processing",
  description="Upload a video or use your webcam for processing."
)

interface.launch(debug = True)
