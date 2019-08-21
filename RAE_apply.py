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
FILEWRITER_PATH = './RAE_tensorboard'
if not os.path.isdir(FILEWRITER_PATH):
    os.makedirs(FILEWRITER_PATH)
CHECKPOINT_PATH = './RAE_tensorboard/checkpoints'
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

# Set path for folder containing the dataset
data_path = glob.glob('C:/Users/Joon/PycharmProjects/PPG_denoising/PPG_1000samples.txt')

# Autoencoder parameters
signal_length = 250
num_hidden_nodes = 40

# Training Parameters
batch_size = 10000
val_batch_size = 10000

# ------------------------------------------------- loading data ----------------------------------------------------- #
# define function to load the data
def load_data(data_path, signal_length):

    # load the dataset
    data = np.loadtxt(data_path)

    # shuffle the order of the samples in the dataset
    shuffle(data)

    # create a list to reorganize the data into segments of None x signal_length shape
    # the original data is in None x 1000 shape
    new_data = []
    for datanum in range(len(data)):
        for index in range(0, 750, 50):
            a = data[datanum, index:index + signal_length].copy()
            a = a - np.min(a)  # normalize the samples
            a = a / np.max(a)
            if np.sum(np.isfinite(a)) == signal_length:  # make sure all the data in the sample are finite
                new_data.append(a)
    data = np.asarray(new_data)
    answer_data = data.copy()

    # #실습8: dataset에 노이즈를 추가해서 denoising autoencoder를 학습시키세요============================================
    # noisy_data = []
    # answer_data = []
    # for datanum in range(len(data)):
    #     dummy = data[datanum, :].copy()
    #     # 노이즈1: 고주파
    #     noise1 =
    #     dummy = dummy + noise1
    #     dummy = dummy - min(dummy)
    #     dummy = dummy / max(dummy)
    #
    #     # 노이즈2: 저주파
    #     noise2 =
    #     for i in range(len(dummy)):
    #         dummy[i] =
    #     dummy = dummy - min(dummy)
    #     dummy = dummy / max(dummy)
    #
    #     # 노이즈3: saturation
    #     location1 =
    #     location2 =
    #     if location2 > signal_length:
    #         location2 = signal_length
    #     dummy[location1:location2] =
    #
    #     noisy_data.append(dummy)
    #     answer_data.append(data[datanum, :])
    #
    # data = np.asarray(noisy_data)
    # answer_data = np.asarray(answer_data)

    # create empty lists for training and validation data
    train_input_data_list = []
    train_output_data_list = []
    val_input_data_list = []
    val_output_data_list = []

    # go through the entire dataset and allocate 80% to training and 20% to validation
    for datanum in range(len(data)):
        if datanum < 0.8*len(data):
            train_input_data_list.append(data[datanum])
            train_output_data_list.append(answer_data[datanum])
        else:
            val_input_data_list.append(data[datanum])
            val_output_data_list.append(answer_data[datanum])

    return train_output_data_list, train_input_data_list, val_output_data_list, val_input_data_list


# load the data into training and validation sets
[train_output, train_input, val_output, val_input] = load_data(data_path[0], signal_length)

print('data loading completed! %i training data and %i validation data' % (len(train_input), len(val_input)))

# ------------------------------------ defining graph input and network structure ------------------------------------ #

# 실습9: autoencoder를 recurrent autoencoder로 변경해보세요 ===========================================================

print(np.shape(train_input))
train_input = np.reshape(train_input, [len(train_input), signal_length, 1])
print(np.shape(train_input))
val_input = np.reshape(val_input, [len(val_input), signal_length, 1])
prob = tf.placeholder_with_default(1.0, shape=())
X = tf.placeholder("float", [None, signal_length, 1])
Y = tf.placeholder("float", [None, signal_length])

def RAE(x, probability, num_hidden_nodes):
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    # before unstacking shape = batch_size, timesteps, num_input
    x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
    # x = tf.unstack(x, signal_length, 1)
    # after unstacking shape = timesteps, batch_size, num_input

    lstm_fw_cell = rnn.LSTMCell(num_hidden_nodes, forget_bias=1.0)
    lstm_fw_cell = rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=probability)

    outputs, _ = rnn.static_rnn(lstm_fw_cell, x, dtype=tf.float32)
    print(outputs[-1].get_shape())

    # Linear activation, using rnn inner loop last output
    logit = tf.layers.dense(outputs[-1], signal_length, activation=None,
                            use_bias=True, name='output_layer',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            bias_initializer=tf.ones_initializer())
    print(logit.get_shape())

    return logit

# define 'prediction' as the output of the autoencoder
prediction = RAE(X, prob, num_hidden_nodes)

# define 'loss' as L2 loss function
loss = tf.losses.mean_squared_error(Y, prediction)

# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# calculate total batch size based on size of train and validation sets
total_batch = len(train_input) // batch_size
total_val_batch = len(val_input) // val_batch_size
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

    # pick a goal for the loss value
    val_loss_epochs = []
    val_loss = 0
    # set a for-loop to go through the validation set
    for j in range(total_val_batch):

        batch_index1 = j * val_batch_size

        val_batch = val_input[batch_index1:batch_index1 + val_batch_size]
        val_label = val_output[batch_index1:batch_index1 + val_batch_size]

        loss2, output_val = sess.run([loss, prediction], feed_dict={X: val_batch, Y: val_label})
        val_loss += loss2 / total_val_batch

    val_loss_epochs.append(val_loss)

    output_val = sess.run(prediction, feed_dict={X: val_batch, Y: val_label})
    random_index = np.random.randint(0, len(val_batch))
    figure = plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(output_val[random_index], color='g')
    plt.plot(val_label[random_index], color='b')
    plt.plot(val_batch[random_index], color='k')
    plt.xlabel('samples')
    plt.ylabel('PPG')

    val_loss_plot = np.asarray(val_loss_epochs)
    plt.subplot(1, 2, 2)
    plt.plot(val_loss_plot, color='b')
    plt.savefig('PPG_RAE_result_example_'+str(signal_length)+'samples_'+str(num_hidden_nodes)+'.png')

    sess.close()