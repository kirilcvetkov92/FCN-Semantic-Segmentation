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

def get_label_info():
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!
    # Arguments
        csv_path: The file path of the class dictionairy

    # Returns
        Two lists: one for the class names and the other for the label values
    """
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

        Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),

        Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    ]

    class_names = []
    label_values = []
    for label in labels:
        class_names.append(label[0])
        label_values.append(label[7])

    return class_names, label_values


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """

    semantic_map = []
    c = np.logical_and(np.not_equal(label, label_values[0]), np.not_equal(label, label_values[1]))
    mask = np.any(c, axis=-1)
    semantic_map.append(mask)
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """

    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def load_image(path):
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    return image


def load_annotation(path):
    image = cv2.imread(path, -1)
    return image


def get_data():
    PATH = 'D:/data/leftImg8bit_trainvaltest/'
    train_path = PATH + '/leftImg8bit/train/'
    trainy_path = PATH + '/sky-data/train/'

    val_path = PATH + '/leftImg8bit/val/'
    valy_path = PATH + '/sky-data/val/'

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
        x = cv2.resize(x, dsize=(256, 512))
        X_train.append(x)

    print('Loading Y_Train..')
    for sample in trainy_batch:
        img_path = sample
        x = load_image(img_path)
        x = cv2.resize(x, dsize=(256, 512))
        y_train.append(x)

    print('Loading X_Validation..')
    for sample in val_batch:
        img_path = sample
        x = load_image(img_path)
        x = cv2.resize(x, dsize=(256, 512))
        X_val.append(x)

    print('Loading Y_Validation..')
    for sample in valy_batch:
        img_path = sample
        x = load_image(img_path)
        x = cv2.resize(x, dsize=(256, 512))
        y_val.append(x)

    return X_train, y_train, X_val, y_val




def flip_image(image, measurement, flip_probability=1.0):
    if random.random() <= flip_probability:
        image = cv2.flip(image, 1)
        measurement*=-1
    return image, measurement


def data_augmentation(input_image, output_image):
    # Data augmentation
    # go here
    if random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if  random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)

    #brightness
    if random.randint(0,1):
        factor = 1.0 + random.uniform(-1.0*0.5, 0.5)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)

    if  random.randint(0,1):
        angle = random.uniform(-1*45, 45)
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

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

    if(batch_size==-1):
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
