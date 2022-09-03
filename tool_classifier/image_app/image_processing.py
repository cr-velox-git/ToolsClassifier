import os
import string
import cv2
import imghdr
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy


def create_model():
    # limiting gpu usage
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    #

    # data_dir = 'data'
    # os.listdir(os.path.join(data_dir,'missing'))
    image_exts = ['jpg', 'jpeg']

    # load data
    # creating a data pipline

    data = tf.keras.utils.image_dataset_from_directory('../processor/data')
    # allowing to loop through data
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    # Images represented as numpy array
    batch0 = batch[0].shape
    print(batch0)

    # PREPROCESSING DATA
    scaled = batch[0] / 255
    data = data.map(lambda x, y: (x / 255, y))  # x - image
    scaled_iterator = data.as_numpy_iterator()
    batch = scaled_iterator.next()

    # to visualize data
    # fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    # for idx, img in enumerate(batch[0][:4]):
    #     ax[idx].imshow(img)
    #     ax[idx].title.set_text(batch[1][idx])

    # splitting the data into train and test
    print('length of data: %d', len(data))

    train_size = int(len(data) * .7)
    val_size = int(len(data) * .2) + 1
    test_size = int(len(data) * .1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size).take(test_size)

    # Building Deep Learning Model

    # creating arc
    model = Sequential()
    # adding a convolution input layer and a max-pooling layer
    # 16 filter of 3*3 size
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # optimizer
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    # TRAIN
    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=18, validation_data=val, callbacks=[tensorboard_callback])

    # plotting performance

    # loss
    # fig = plt.figure()
    # plt.plot(hist.history['loss'], color='teal', label='loss')
    # plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    # fig.suptitle('Loss', fontsize=20)
    # plt.legend(loc="upper left")
    # plt.show()
    #
    # # accuracy
    # fig = plt.figure()
    # plt.plot(hist.history['accuracy'], color='teal', label='acuracy')
    # plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    # fig.suptitle('Accuracy', fontsize=20)
    # plt.legend(loc="upper left")
    # plt.show()

    # EVALUTE PERFORMANCE

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test.as_numpy_iterator():
        x, y = batch
        yhat = model.predict(x)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

    # testing a image
    # img = cv2.imread('test_image_al.jpg')
    # plt.imshow(img)
    # plt.show()
    # resize = tf.image.resize(img, (256, 256))
    # plt.imshow(resize.numpy().astype(int))
    # plt.show()

    # yhat = model.predict(np.expand_dims(resize / 255, 0))

    # if yhat > 0.5:
    #     print(f'predicted some tools are missing')
    # else:
    #     print(f'predicted all tools are available')

    # SAVE THE MODEL

    model.save(os.path.join('models', 'tools_model.h5'))
    print('model saved')


def test_image(image_loc: string):
    img = mpimg.imread(image_loc)
    plt.imshow(img)
    # plt.show()

    resize = tf.image.resize(img, (256, 256))
    plt.imshow(resize.numpy().astype(int))
    # plt.show()

    print('Loading the model')
    new_model = load_model(os.path.join('../processor/models', 'tools_model.h5'))
    yhatnew = new_model.predict(np.expand_dims(resize / 255, 0))
    print('Model Predicted')
    if yhatnew > 0.5:
        # print(f'predicted some tools are missing')
        return 0
    else:
        # print(f'predicted all tools are available')
        return 1