""" Biosignal autoencoder practice
    
    KOSOMBE 2019 Summer School of Biomedical Engineering
    Deep learning based signal processing: autoencoders & recurrent neural networks

    Author: Joonnyong Lee (Seoul National University Hospital, Biomedical Research Institute)
    Date: 2019-8-24
"""

# ----------------------------------------------------- setup -------------------------------------------------------- #
# import all the necessary libaries
from random import shuffle
import glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import matplotlib.pyplot as plt

# Set paths for tensorboard and checkpoints
FILEWRITER_PATH = './BRAE_tensorboard'
if not os.path.isdir(FILEWRITER_PATH):
    os.makedirs(FILEWRITER_PATH)
CHECKPOINT_PATH = './BRAE_tensorboard/checkpoints'
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

# Set path for folder containing the dataset
data_path = glob.glob('C:/Users/Joon/PycharmProjects/PPG_denoising/*file.txt')

# Autoencoder parameters
signal_length = 250
num_hidden_nodes = 40

# 이론에서 발표 드렸듯이, 250샘플의 PPG를 50샘플 마다 겹치게 해서 노이즈를 제거하고 평균을 내는 작업을 위한 변수 (이론 자료 p112)
step_size = 50

# ------------------------------------------------- loading data ----------------------------------------------------- #
# define function to load the data
def load_data(data_file, signal_length):
    # load the text data file
    data = np.loadtxt(data_file)
    name = data_file[:len(data_file)-4]
    name = name + '_BRAE_denoised.txt'
    data_list = []
    for datanum in range(0, len(data) - signal_length, step_size):
        dummy = data[datanum:datanum + signal_length]
        dummy = dummy - min(dummy)
        dummy = dummy / max(dummy)
        data_list.append(dummy)

    return data_list, data, name

# 이론실습 p112에서 말하는 작업을하는 함수
def overlap_averaging(input_data):
    data_list2 = []
    for datanum2 in range(1, len(input_data)):
        if datanum2 < signal_length/step_size:
            dummy = []
            for overlapnum in range(datanum2):
                dummy.append(input_data[overlapnum, (datanum2 - overlapnum) * step_size : (datanum2 - overlapnum + 1) * step_size])
            data_list2.append(np.mean(np.asarray(dummy), axis=0))
        else:
            dummy = []
            for overlapnum in range(int(signal_length/step_size)):
                dummy.append(input_data[datanum2 - int(signal_length/step_size) + overlapnum, signal_length - step_size * (overlapnum+1) : signal_length - step_size * overlapnum])
            data_list2.append(np.mean(np.asarray(dummy), axis=0))

    data_list2 = np.reshape(data_list2, [-1])

    smoothed_data = data_list2.copy()
    smoothing_window_size = 5
    for index in range(smoothing_window_size, len(data_list2)-smoothing_window_size):
        dummy = data_list2[index-smoothing_window_size:index+smoothing_window_size].copy()
        dummy.sort()
        smoothed_data[index] = np.mean(dummy[1:len(dummy)-1])

    return smoothed_data

# ------------------------------------ defining graph input and network structure ------------------------------------ #
prob = tf.placeholder_with_default(1.0, shape=())
X = tf.placeholder("float", [None, signal_length, 1])
Y = tf.placeholder("float", [None, signal_length])

def BRAE(x, probability, num_hidden_nodes):
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    # before unstacking shape = batch_size, timesteps, num_input
    x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
    # x = tf.unstack(x, signal_length, 1)
    # after unstacking shape = timesteps, batch_size, num_input

    lstm_fw_cell = rnn.LSTMCell(num_hidden_nodes, forget_bias=1.0)
    lstm_fw_cell = rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=probability)
    lstm_bw_cell = rnn.LSTMCell(num_hidden_nodes, forget_bias=1.0)
    lstm_bw_cell = rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=probability)

    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    print(outputs[-1].get_shape())

    # Linear activation, using rnn inner loop last output
    logit = tf.layers.dense(outputs[-1], signal_length, activation=None,
                            use_bias=True, name='output_layer',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            bias_initializer=tf.ones_initializer())
    print(logit.get_shape())

    return logit

# define 'prediction' as the output of the autoencoder
prediction = BRAE(X, prob, num_hidden_nodes)

# Initialize the variables (i.e. assign their default value)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# 이 실습은 이미 학습된 모델을 가져와서 실제 PPG의 잡음을 없에는 것이기 때문에 학습 (training) 또 학습 검증 (validation)에 필요한 모든 것은 없에도 됩니다.
# 따라서 loss, optimizer, total_batch, batch size 이런 변수들을 다 제거하였습니다

# ---------------------------------------------------- session ------------------------------------------------------- #
with tf.Session() as sess:
    # run the initializer
    sess.run(init)
    # Writer
    writer = tf.summary.FileWriter(FILEWRITER_PATH)
    # Saver
    saver = tf.train.Saver(max_to_keep=200)
    # restore checkpoint ==============================================================================================
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print("loading the lastest checkpoint: " + latest_ckpt)
    saver.restore(sess, latest_ckpt)

    for filenum in range(len(data_path)):
        print(data_path[filenum])
        [input_data, raw_data, name] = load_data(data_path[filenum], signal_length)
        input_data2 = np.reshape(input_data, [len(input_data), signal_length, 1])
        output_val = sess.run(prediction, feed_dict={X: input_data2})

        # 이론실습 p112에서 말하는 작업을하는 함수
        save_data = overlap_averaging(output_val)

        # save prediction results
        file = open(name, 'w')
        for j in range(len(save_data)):
            file.write("%f " % save_data[j])
        file.close()

        # 처음 10초만 예제로 plot함
        figure = plt.figure(figsize=(14, 6))
        plt.plot(save_data[:signal_length*5], color='g')
        plt.plot(raw_data[:signal_length*5], color='b')
        plt.xlabel('time')
        plt.ylabel('PPG')
        plt.savefig('PPG_BRAE_example_' + str(filenum) + '.png')

    sess.close()
