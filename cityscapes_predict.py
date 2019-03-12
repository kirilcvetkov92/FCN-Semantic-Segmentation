#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import cityscapes_helper
import numpy as np

label_values = cityscapes_helper.get_label_info()

class_names, label_values = cityscapes_helper.get_label_info()
L2_REG = 1e-6
STDEV = 1e-3
KEEP_PROB = 0.5
LEARNING_RATE = 1e-4


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):


    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    labels = tf.reshape(correct_label, (-1, num_classes), name='labels')
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    softmax = tf.nn.softmax(logits, name='softmax')

    train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cross_entropy_loss)


    return logits, train_op, cross_entropy_loss


def pipeline_final(img, is_video):
    channel = 1 if is_video else 4
    size = img.shape
    img = cv2.resize(img, dsize=(512, 256))
    img = np.array([img])
    softmax_ = loss = sess.run([softmax],
                               feed_dict={input: img, keep_prob: 1})
    logits_ = (softmax_[0].reshape(1, 256, 512, 29))
    output_image = reverse_one_hot(logits_[0])

    print(output_image.shape)

    out_vis_image = colour_code_segmentation(output_image, c_values)

    a = cv2.cvtColor(np.uint8(out_vis_image), channel)

    b = cv2.cvtColor(np.uint8(img[0]), channel)

    added_image = cv2.addWeighted(a, 1, b, 1, channel)
    added_image = cv2.resize(added_image, dsize=(512, 256))

    return added_image

def pipeline_video(img):
    return pipeline_final(img, True)

def pipeline_img(img):
    return pipeline_final(img, False)

def process(media_dir, save_dir, model_dir):
    global model, label_values

    data_dir = './data'
    num_classes = 29
    tf.reset_default_graph()
    sess = tf.Session()

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    # Create function to get batches
    get_batches_fn = gen_batch_function

    input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
    print(input)
    output = layers(layer3, layer4, layer7, num_classes)

    tf.set_random_seed(123)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()  # Simple model saver

    saver.restore(sess, tf.train.latest_checkpoint('.'))

    logits = tf.reshape(output, (-1, num_classes))
    softmax = tf.nn.softmax(logits, name='softmax')



    model = load_model(model_dir, custom_objects={'preprocess_input': preprocess_input})
    label_values, _, _ = get_label_values()

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
                          model_dir = parsed_arg.model_dir
                          )