#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import cityscapes_helper
from cityscapes_config import *


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph()
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return input, keep_prob, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    l7_conv = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer= tf.random_normal_initializer(stddev=STDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG), name='conv_1_1_1',activation = tf.nn.relu)


    conv1 = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, 4, 2,
                                        padding='same', kernel_initializer= tf.random_normal_initializer(stddev=STDEV),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG), name='conv_1_1_2',activation = tf.nn.relu)

    l4_conv = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer= tf.random_normal_initializer(stddev=STDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG), name='conv_1_1_3',activation = tf.nn.relu)

    skip_1 = tf.add(conv1, l4_conv, name='conv_1_1_4')
    #output = tf.layers.batch_normalization(output)
    #output = keras.layers.UpSampling2D(size=(2,2),data_format=None,interpolation='bilinear')(output)

    conv2 = tf.layers.conv2d_transpose(skip_1, num_classes, 4, 2,
                                       padding='same', kernel_initializer= tf.random_normal_initializer(stddev=STDEV),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),  name='conv_1_1_5',activation = tf.nn.relu)
    l3_conv = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer= tf.random_normal_initializer(stddev=STDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG), name='conv_1_1_6',activation = tf.nn.relu)
    skip_3 = tf.add(conv2, l3_conv,  name='conv_1_1_7')

    output = tf.layers.conv2d_transpose(skip_3, num_classes, 16, 8,
                                        padding='same', kernel_initializer= tf.random_normal_initializer(stddev=STDEV),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),  name='conv_1_1_8',activation = tf.nn.relu)

    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    labels = tf.reshape(correct_label, (-1, num_classes), name='labels')
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)

import time


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate, X_train, y_train, label_values, X_val, y_val):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    for epoch in range(epochs):
        print('epoch : ', epoch)
        for image, targets in get_batches_fn(X_train, y_train, label_values, batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: targets, keep_prob: KEEP_PROB,
                                          learning_rate: LEARNING_RATE})
        # Print data on the learning process

            print("Epoch: {}".format(epoch + 1), "/ {}".format(epochs), " Loss: {:.3f}".format(loss))

        mean_loss = []
        for image, targets in get_batches_fn(X_val, y_val, label_values, batch_size, is_train=False):
            loss = sess.run([cross_entropy_loss],
                            feed_dict={input_image: image, correct_label: targets, keep_prob: 1})
            mean_loss.append(loss)

        mean_loss_ = np.mean(np.array(mean_loss))
        print("Epoch: {}".format(epoch + 1), "/ {}".format(epochs), " Validation Loss: {:.3f}".format(mean_loss_))

        print('saving model')
        saver.save(sess, './model')


#tests.test_train_nn(train_nn)


def run():
    num_classes = NUM_CLASSES
    image_shape = (512, 256)  # Cityscapes dataset should be scaled (Using GTX-1080TI)
    data_dir = DATA_DIR
    video_dir = VIDEO_DIR
    runs_dir = RUNS_DIR

    epochs = EPOCHS
    batch_size = BATCH_SIZE
    is_train = IS_TRAIN
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    label_values = cityscapes_helper.get_label_info()

    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    print('run')
    tf.reset_default_graph()
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = cityscapes_helper.gen_batch_function

        input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        print(input)
        output = layers(layer3, layer4, layer7, num_classes)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)
        logits = tf.nn.softmax(logits, name='softmax')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # Simple model saver

        if not is_train:
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            cityscapes_helper.save_inference_samples(runs_dir, video_dir, sess, image_shape, logits, keep_prob, input, label_values)
        else:
            X_train, y_train, X_val, y_val = cityscapes_helper.get_data(data_dir)
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input, correct_label,
                         keep_prob, learning_rate, X_train, y_train, label_values, X_val, y_val)
            cityscapes_helper.save_inference_samples(runs_dir, video_dir, sess, image_shape, logits, keep_prob, input,
                                                     label_values)
            saver.save(sess, './model')

if __name__ == '__main__':
    run()