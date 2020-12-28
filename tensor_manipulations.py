#------------------------------------------------#
# Title:                            <some title>
# Author:                            Stanislav Sys
# Date:                                 22.12.2020
#------------------------------------------------#

# Imports
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
# load data
(train_images,train_labels), (test_images,test_labels) = mnist.load_data()

def main():

    print(f'number of dimension in train_images tensor is: {train_images.ndim}')

    print(f'The shape of the data tensor is {train_images.shape}')

    print(f'The type of da is: {train_images.dtype}')

    #plt.imshow(train_images[4], cmap=plt.cm.binary)
    #plt.show()

    #doing operations on tensors
    my_slice =train_images[10:100, :, :]
    print(f'shape of my slice: {my_slice.shape}')

    # reshaping tensors

    x = np.array([[0., 1.],
                 [2., 3.],
                 [4., 6.]])
    print(x)
    print(f'Shape of my numpy tensor is: {x.shape}')

    x = x.reshape((6,1))
    print(f'Shape of my reshaped data: {x.shape}')
    print(x)









# Main Method

if __name__ == "__main__":
    main()


