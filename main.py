#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import cityscapes_helper


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

import keras
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
    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

   # layer7_conv_1x1 = tf.layers.batch_normalization(layer7_conv_1x1)

    output = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, 4, 2,
                                        padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    #output = keras.layers.UpSampling2D(size=(2,2),data_format=None,interpolation='bilinear')(layer7_conv_1x1)

    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    #layer4_conv_1x1 = tf.layers.batch_normalization(layer4_conv_1x1)

    output = tf.add(output, layer4_conv_1x1)
   # output = tf.layers.batch_normalization(output)

    #output = keras.layers.UpSampling2D(size=(2,2),data_format=None,interpolation='bilinear')(output)


    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2,
                                       padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.add(output, layer3_conv_1x1)
  #  output = tf.layers.batch_normalization(output)

    #output = keras.layers.UpSampling2D(size=(8,8),data_format=None,interpolation='bilinear')(output)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8,
                                        padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

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

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)



def validate_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    # TODO: Implement function
    # TODO: Implement function
    for image, targets in get_batches_fn(batch_size):
        _, loss = sess.run([cross_entropy_loss],
                           feed_dict={input_image: image, correct_label: targets, keep_prob: 0.5,
                                      learning_rate: 0.001})
    # Print data on the learning process
    print("Validation: {}  Loss: {:.3f}".format(loss))



import time
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'

    class_names, label_values = cityscapes_helper.get_label_info()
    X_train, y_train, X_val, y_val = cityscapes_helper.get_data()

    # TODO: Implement function
    # TODO: Implement function
    for epoch in range(epochs):
        for image, targets in get_batches_fn(X_train, y_train, X_val, y_val, label_values, batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: targets, keep_prob: 0.5,
                                          learning_rate: 0.001})
        # Print data on the learning process

            print("Epoch: {}".format(epoch + 1), "/ {}".format(epochs), " Loss: {:.3f}".format(loss))

        # validate_nn(sess, epochs, 1, get_batches_fn_valid, train_op, cross_entropy_loss, input, correct_label,
        #             keep_prob, learning_rate)

tests.test_train_nn(train_nn)

def run():
    num_classes = 3
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = 100
    batch_size = 20
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto()) as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = cityscapes_helper.gen_batch_function

        input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output = layers_cityscapes(layer3, layer4, layer7, num_classes)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)
        tf.set_random_seed(123)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # Simple model saver


        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input, correct_label,
                 keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input)
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
