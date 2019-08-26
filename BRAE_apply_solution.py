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
        for index in range(0, 1000-signal_length, 50):
            a = data[datanum, index:index + signal_length].copy()
            a = a - np.min(a)  # normalize the samples
            a = a / np.max(a)
            if np.sum(np.isfinite(a)) == signal_length:  # make sure all the data in the sample are finite
                new_data.append(a)
    data = np.asarray(new_data)
    answer_data = data.copy()

    #실습8: dataset에 노이즈를 추가해서 denoising autoencoder를 학습시키세요============================================
    noisy_data = []
    answer_data = []
    for datanum in range(len(data)):
        for generation_num in range(1):
            dummy = data[datanum, :].copy()
            # high frequency noise
            noise1 = np.random.randn(signal_length) / 3
            dummy = dummy + noise1
            dummy = dummy - min(dummy)
            dummy = dummy / max(dummy)

            # sloping noise
            noise2 = (np.random.rand(1) - 0.5) * -4
            for i in range(len(dummy)):
                dummy[i] = dummy[i] + noise2 * i / len(dummy)
            dummy = dummy - min(dummy)
            dummy = dummy / max(dummy)

            # saturation noise
            location1 = int(np.floor(np.random.rand(1) * signal_length))
            location2 = location1 + int(np.floor(np.random.rand(1) * signal_length / 5))
            if location2 > signal_length:
                location2 = signal_length
            dummy[location1:location2] = np.round(np.random.rand(1)) * np.ones((location2 - location1), float)

            noisy_data.append(dummy)
            answer_data.append(data[datanum, :])

    data = np.asarray(noisy_data)
    answer_data = np.asarray(answer_data)

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

# 실습10: autoencoder를 bidirectional recurrent autoencoder로 변경해보세요 ============================================

print(np.shape(train_input))
train_input = np.reshape(train_input, [len(train_input), signal_length, 1])
print(np.shape(train_input))
val_input = np.reshape(val_input, [len(val_input), signal_length, 1])
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
    old_loss = 0.001
    train_loss_epochs = []
    train_loss_epochs2 = []
    val_loss_epochs = []
    # set the number of iterations (epochs) for training & validation
    for epoch in range(300):
        # shuffle the training data at the beginning of each iteration for better generalization
        if 1:
            d = list(zip(train_input, train_output))
            shuffle(d)
            train_input, train_output = zip(*d)
        # create a variable to hold training loss value after each epoch
        train_loss = 0
        # set a for-loop to go through the training set
        for i in range(total_batch):

            # set index for training data batch
            batch_index = i * batch_size

            # extract training data batch
            batch = train_input[batch_index:batch_index + batch_size]
            label = train_output[batch_index:batch_index + batch_size]

            # run optimizer and calculate loss
            _, loss1 = sess.run([optimizer, loss], feed_dict={X: batch, Y: label, prob: 0.5})

            # add loss at end of batch to the total training loss for this epoch
            train_loss += loss1/total_batch

        train_loss_epochs.append(train_loss)

        # 실습1: training과 비슷하게 validation을 만드세요 ============================================================
        # validation 할때는 dropout을 끄세요
        # create a variable to hold validation loss value after each epoch
        val_loss = 0
        # set a for-loop to go through the validation set
        for j in range(total_val_batch):

            batch_index1 = j * val_batch_size

            val_batch = val_input[batch_index1:batch_index1 + val_batch_size]
            val_label = val_output[batch_index1:batch_index1 + val_batch_size]

            loss2, output_val = sess.run([loss, prediction], feed_dict={X: val_batch, Y: val_label})
            val_loss += loss2 / total_val_batch

        val_loss_epochs.append(val_loss)

        # after every 5 epochs, create a figure to save the output of the autoencoder
        if epoch % 5 == 0:
            output_val = sess.run(prediction, feed_dict={X: val_batch, Y: val_label})
            random_index = np.random.randint(0, len(val_batch))
            figure = plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.plot(output_val[random_index], color='g')
            plt.plot(val_label[random_index], color='b')
            plt.plot(val_batch[random_index], color='k')
            plt.xlabel('samples')
            plt.ylabel('PPG')

            train_loss_plot = np.asarray(train_loss_epochs)
            val_loss_plot = np.asarray(val_loss_epochs)
            plt.subplot(1, 2, 2)
            plt.plot(train_loss_plot, color='g')
            plt.plot(val_loss_plot, color='b')
            plt.savefig('PPG_RAE_result_example_'+str(signal_length)+'samples_'+str(num_hidden_nodes)+'nodes_'+str(epoch)+'_epoch.png')

        # save the neural network state if the loss value is lower than the goal
        if 1:  # val_loss < old_loss:
            # update the goal for the loss value
            old_loss = val_loss
            # save checkpoint
            checkpoint_name = os.path.join(CHECKPOINT_PATH, 'PPG_BRAE_' + str(num_hidden_nodes) + 'nodes_' + str(
                                               signal_length) + 'sample_epoch')
            save_path = saver.save(sess, checkpoint_name, global_step=epoch)

        # print the result of the optimization after each epoch
        print('-----------------------------------------------------------------------------')
        print("Epoch: %d " % epoch)
        print('training loss: %f' % train_loss)
        print('val loss: %f' % val_loss)
        print('min loss: %f' % old_loss)

    sess.close()