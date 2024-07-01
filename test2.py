import tensorflow as tf
from keras.models import load_model
import gradio as gr
import os
import cv2

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
model= load_model('D:\Other Files\client_projects\LipReading\LipNet.h5', custom_objects={'CTCLoss': CTCLoss})
print(model.summary())

def predict(video):
    a,b,c,d = cropping_dictionary['s1']
    cap = cv2.VideoCapture(video)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[a:b,c:d,:])
    cap.release()
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    frames = tf.cast((frames - mean), tf.float32) / std
    frames = tf.expand_dims(frames,axis = 0)
    yhat = model.predict(frames)
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    output= [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded] 
    print(output)

predict('D:\Other Files\client_projects\LipReading-1\bbaf2n.mpg')