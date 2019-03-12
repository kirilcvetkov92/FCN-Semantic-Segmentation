#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import cityscapes_helper
import numpy as np
import argparse as parser
from cityscapes_config import *
from cityscapes_helper import *
from moviepy.editor import VideoFileClip
import cv2

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    return image


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


    conv1 = tf.layers.conv2d_transpose(l7_conv, num_classes, 4, 2,
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


def pipeline_final(img, is_video):
    channel = 1 if is_video else 4

    img = cv2.resize(img, dsize=(512, 256))
    img = np.array([img])
    softmax_   = sess.run([softmax],
                               feed_dict={input: img, keep_prob: 1})
    logits_ = (softmax_[0].reshape(1, 256, 512, 29))
    output_image = reverse_one_hot(logits_[0])

    print(output_image.shape)

    out_vis_image = colour_code_segmentation(output_image, label_values)

    a = cv2.cvtColor(np.uint8(out_vis_image), channel)

    b = cv2.cvtColor(np.uint8(img[0]), channel)

    added_image = cv2.addWeighted(a, 1, b, 1, channel)
    added_image = cv2.resize(added_image, dsize=(512, 256))

    return added_image

def pipeline_video(img):
    return pipeline_final(img, True)

def pipeline_img(img):
    return pipeline_final(img, False)

def process(media_dir, save_dir):
    global sess, softmax, label_values, input, keep_prob

    data_dir = './data'
    num_classes = 29

    label_values = cityscapes_helper.get_label_info()

    tf.reset_default_graph()
    sess = tf.Session()

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    # Create function to get batches

    input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
    output = layers(layer3, layer4, layer7, num_classes)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()  # Simple model saver

    saver.restore(sess, tf.train.latest_checkpoint('.'))

    logits = tf.reshape(output, (-1, num_classes))
    softmax = tf.nn.softmax(logits, name='softmax')


    try:
        img = load_image(media_dir)
        output = os.path.join(save_dir, 'output_image.png')
        img = pipeline_img(img)
        cv2.imwrite(output, img)
    except Exception as ex:
        output = os.path.join(save_dir, 'output_video.mp4')
        clip1 = VideoFileClip(media_dir)
        white_clip = clip1.fl_image(pipeline_video)
        white_clip.write_videofile(output, audio=False)

if __name__ == '__main__':

    if __name__ == "__main__":
        args = parser.ArgumentParser(description='Model prediction arguments')

        args.add_argument('-media', '--media_dir', type=str,
                          help='Media Directorium for prediction (mp4,png)')

        args.add_argument('-save', '--save_dir', type=str, default=DEFAULT_SAVE_DIR,
                          help='Save Directorium')

        args.add_argument('-model', '--model_dir', type=str, default=PRETRAINED_MODEL_DIR,
                          help='Model Directorium')

        parsed_arg = args.parse_args()

        crawler = process(media_dir=parsed_arg.media_dir,
                          save_dir=parsed_arg.save_dir,
                          )