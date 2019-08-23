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
FILEWRITER_PATH = './AE_tensorboard'
if not os.path.isdir(FILEWRITER_PATH):
    os.makedirs(FILEWRITER_PATH)
CHECKPOINT_PATH = './AE_tensorboard/checkpoints'
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

# Set path for folder containing the dataset
data_path = glob.glob('C:/Users/Joon/PycharmProjects/PPG_denoising/PPG_1000samples.txt')

# Autoencoder parameters
signal_length = 250
num_hidden_nodes = 40

# Training Parameters
batch_size = 9000
val_batch_size = 9000

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
# define placeholders
prob = tf.placeholder_with_default(1.0, shape=())
is_training = tf.placeholder_with_default(False, shape=())
X = tf.placeholder("float", [None, signal_length])
Y = tf.placeholder("float", [None, signal_length])

# 실습6: autoencoder를 5 layer로 변경해보세요 ==========================================================================
# define autoencoder as a function
def AE(x, probability, num_hidden_nodes, training_true):
    # define hidden layer
    # 실습5: hidden layer의 노드수와 activation function을 변경해보세요 ================================================
    encode = tf.layers.dense(x, num_hidden_nodes, activation=tf.nn.sigmoid,
                          use_bias=True, name='encoding_layer',
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          bias_initializer=tf.ones_initializer())
    encode = tf.layers.dropout(encode, rate=probability, training=training_true)
    print(encode.get_shape())

    # define output layer
    # 실습4: output layer에 activation function을 추가해보세요 =========================================================
    logit = tf.layers.dense(encode, signal_length, activation=None,
                             use_bias=True, name='output_layer',
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             bias_initializer=tf.ones_initializer())
    print(logit.get_shape())

    return logit

# define 'prediction' as the output of the autoencoder
prediction = AE(X, prob, num_hidden_nodes, is_training)

# 실습3: loss function을 L1으로 변경하고 학습 결과를 비교하세요 ======================================================
# define 'loss' as L2 loss function
loss = tf.losses.mean_squared_error(Y, prediction)

# 실습2: optimizer를 변경하고 학습 결과를 비교하세요 =================================================================
# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

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

    # pick a goal for the loss value
    old_loss = 0.03
    train_loss_epochs = []
    val_loss_epochs = []
    # set the number of iterations (epochs) for training & validation
    for epoch in range(300):
        # shuffle the training data at the beginning of each iteration for better generalization
        if True:
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
            _, loss1 = sess.run([optimizer, loss], feed_dict={X: batch, Y: label, prob: 0.5, is_training: True})

            # add loss at end of batch to the total training loss for this epoch
            train_loss += loss1/total_batch

        train_loss_epochs.append(train_loss)

        # 실습1: training과 비슷하게 validation을 만드세요 ============================================================
        # validation 할때는 dropout을 끄세요


        """
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
            plt.savefig('PPG_AE_result_example_'+str(signal_length)+'samples_'+str(num_hidden_nodes)+'nodes_'+str(epoch)+'_epoch.png')

        # save the neural network state if the loss value is lower than the goal
        if val_loss < old_loss:
            # update the goal for the loss value
            old_loss = val_loss

            # save checkpoint
            checkpoint_name = os.path.join(CHECKPOINT_PATH, 'PPG_AE_' + str(num_hidden_nodes) + 'nodes_' + str(
                                               signal_length) + 'sample_epoch')
            save_path = saver.save(sess, checkpoint_name, global_step=epoch)
        """
        # 실습7: 0, 0.5, 1로만 구성된 1x250 데이터를 만들고 autoencoder에 넣어서 출력을 확인해보세요

        # print the result of the optimization after each epoch
        print('-----------------------------------------------------------------------------')
        print("Epoch: %d " % epoch)
        print('training loss: %f' % train_loss)
        # print('val loss: %f' % val_loss)
        # print('min loss: %f' % old_loss)

    sess.close()