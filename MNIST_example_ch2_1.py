#------------------------------------------------#
# Title:                                keras test 
# Author:                            Stanislav Sys
# Date:                                 22.12.2020
#------------------------------------------------#

# Imports
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(f'#---------------------#\n Shape: {train_images.shape}')

    # build the network
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
    network.add(layers.Dense(10, activation='softmax'))

    # compile the network
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # prepare the image data
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # now train the network
    network.fit(train_images, train_labels, epochs = 5, batch_size = 128)

    # perform inference 
    test_loss, test_acc = network.evaluate(test_images, test_labels)





# Main Method

if __name__ == "__main__":
    main()

