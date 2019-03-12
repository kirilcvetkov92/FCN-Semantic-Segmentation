import cv2
from collections import namedtuple
import random
import numpy as np
import os.path
import scipy.misc
import os
from glob import glob
from sklearn.utils import shuffle
import random
import time

def get_label_info():
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!
    # Arguments
        csv_path: The file path of the class dictionairy

    # Returns
        Two lists: one for the class names and the other for the label values
    """

    # a label and all meta information
    Label = namedtuple('Label', [

        'name',  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class

        'id',  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.

        'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!

        'category',  # The name of the category that this label belongs to

        'categoryId',  # The ID of this category. Used to create ground truth images
        # on category level.

        'hasInstances',  # Whether this label distinguishes between single instances or not

        'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not

        'color',  # The color of this label
    ])

    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    seen = set()
    label_list = list(map(lambda x : x[7], labels))
    label_values = [x for x in label_list if not (x in seen or seen.add(x))]

    return label_values

def get_data(data_path):
    train_path = data_path + '/leftImg8bit/train/'
    trainy_path = data_path + '/sky-data/train/'

    val_path = data_path + '/leftImg8bit/val/'
    valy_path = data_path + '/sky-data/val/'

    train_batch = glob(os.path.join(train_path, '*/*.png'))

    trainy_batch = glob(os.path.join(trainy_path, '*/*color.png'))
    val_batch = glob(os.path.join(val_path, '*/*.png'))

    valy_batch = glob(os.path.join(valy_path, '*/*color.png'))

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    print('Loading X_Train..')
    for sample in train_batch:
        img_path = sample
        x = load_image(img_path)
        x = cv2.resize(x, dsize=(512, 256))
        X_train.append(x)

    print('Loading Y_Train..')
    for sample in trainy_batch:
        img_path = sample
        x = load_image(img_path)
        x = cv2.resize(x, dsize=(512, 256))
        y_train.append(x)

    print('Loading X_Validation..')
    for sample in val_batch:
        img_path = sample
        x = load_image(img_path)
        x = cv2.resize(x, dsize=(512, 256))
        X_val.append(x)

    print('Loading Y_Validation..')
    for sample in valy_batch:
        img_path = sample
        x = load_image(img_path)
        x = cv2.resize(x, dsize=(512, 256))
        y_val.append(x)

    return X_train, y_train, X_val, y_val

def one_hot_it(label, label_values):
    semantic_encoding = []
    # c = np.logical_and(np.not_equal(label, label_values[0]), np.not_equal(label, label_values[1]))
    # mask = np.any(c, axis=-1)
    # semantic_map.append(mask)
    for color in label_values:
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_encoding.append(class_map)
    semantic_map = np.stack(semantic_encoding, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):

    label_matrix = np.array(label_values)
    x = label_matrix[image.astype(int)]

    return x


def load_image(path):
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    return image


def load_annotation(path):
    image = cv2.imread(path, -1)
    return image


def flip_image(image, measurement, flip_probability=1.0):
    if random.random() <= flip_probability:
        image = cv2.flip(image, 1)
        measurement *= -1
    return image, measurement


# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1] - crop_width)
        y = random.randint(0, image.shape[0] - crop_height)

        if len(label.shape) == 3:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width, :]
        else:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
            crop_height, crop_width, image.shape[0], image.shape[1]))


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    rand_width_scal_1 = np.random.rand()
    IMAGE_HEIGHT, IMAGE_WIDTH, _ = image.shape
    x1, y1 = IMAGE_WIDTH * rand_width_scal_1, 0
    rand_width_scal_2 = np.random.rand()
    x2, y2 = IMAGE_WIDTH * rand_width_scal_2, IMAGE_HEIGHT
    xn, yn = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    mask = np.zeros_like(image[:, :, 1])
    mask[(yn - y1) * (x2 - x1) - (y2 - y1) * (xn - x1) > 0] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    """

    hsv_channel = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness_scalar = np.random.rand()
    ratio = 1.0 + 0.4 * (brightness_scalar - 0.5)
    hsv_channel[:, :, 2] = hsv_channel[:, :, 2] * ratio
    return cv2.cvtColor(hsv_channel, cv2.COLOR_HSV2RGB)


def rotation(input_image, output_image, degrees):
    angle = random.uniform(-1*degrees, degrees)
    M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
    input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
    output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)
    return input_image, output_image

def brightness(image, bright_0_1):
            factor = 1.0 + random.uniform(-1.0*bright_0_1, bright_0_1)
            table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            image = cv2.LUT(image, table)
            return image


def data_augmentation(input_image, output_image):

    if random.randint(0, 1):
        input_image = brightness(input_image, 0.5)

    if random.randint(0, 1):
        input_image = random_shadow(input_image)


    return input_image, output_image


def gen_batch_function(samplesX, samplesY, label_values, batch_size=1, is_train=True):
    """
    F function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:"""

    samplesX, samplesY = shuffle(samplesX, samplesY)

    # Shuffle training data

    num_samples = len(samplesX)

    if (batch_size == -1):
        batch_size = num_samples

    # Loop through batches and grab images, yielding each batch
    for batch_i in range(0, num_samples, batch_size):

        X_train = samplesX[batch_i:batch_i + batch_size]
        y_train = samplesY[batch_i:batch_i + batch_size]

        # preprocessing if required
        X_f = []
        y_f = []
        for x, y in zip(X_train, y_train):
            if is_train:
                x, y = data_augmentation(x, y)

            y = np.float32(one_hot_it(y, label_values=label_values))
            X_f.append(x)
            y_f.append(y)

        X_f = np.float32(X_f)
        y_f = np.float32(y_f)
        yield X_f, y_f

def pipeline_final(img, sess, logits, keep_prob, input_image, image_shape, num_classes=29):
    channel = 1 if is_video else 4
    size = img.shape
    img= cv2.resize(img, dsize=image_shape)
    img = np.array([img])
    softmax = tf.nn.softmax(logits, name='softmax')
    softmax_ = loss = sess.run([softmax],
                       feed_dict={input_image: img, keep_prob:1})
    logits_ = (softmax_[0].reshape(1,image_shape[1],image_shape[0],num_classes))
    output_image = reverse_one_hot(logits_[0])

    print(output_image.shape)

    out_vis_image = colour_code_segmentation(output_image, c_values)

    a = cv2.cvtColor(np.uint8(out_vis_image), channel)

    b = cv2.cvtColor(np.uint8(img[0]), channel)

    added_image = cv2.addWeighted(a, 1, b, 1, channel)
    added_image = cv2.resize(added_image, image_shape)

    return added_image


def process(media_dir, sess, logits, keep_prob, input_image, image_shape):
    img = load_image(media_dir)
    img = pipeline_final(img, sess, logits, keep_prob, input_image, image_shape)
    return img


def gen_test_output(sess, logits, keep_prob, input_image, data_folder, image_shape):

    for image_file in glob(os.path.join(data_folder, 'test', '*.png')):
        image = process(image_file, sess, logits, keep_prob, input_image, image_shape)
        yield os.path.basename(image_file), image


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
	"""
	Save test images with semantic masks of lane predictions to runs_dir.
	:param runs_dir: Directory to save output images
	:param data_dir: Path to the directory that contains the datasets
	:param sess: TF session
	:param image_shape: Tuple - Shape of image
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param input_image: TF Placeholder for the image placeholder
	"""
	# Make folder for current run
	output_dir = os.path.join(runs_dir, str(time.time()))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir)

	# Run NN on test images and save them to HD
	print('Training Finished. Saving test images to: {}'.format(output_dir))
	image_outputs = gen_test_output(
		sess, logits, keep_prob, input_image, os.path.join(data_dir, '/leftImg8bit/'), image_shape)
	for name, image in image_outputs:
		scipy.misc.imsave(os.path.join(output_dir, name), image)
