'''
Parameters
'''

import tensorflow as tf
from tf.keras.preprocessing.image import ImageDataGenerator
from tf.keras import callbacks
import os
import argparse
from Model import *

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    #parser.add_argument('-t', '--testing', action='store_true',
    #                    help="Test the trained model on testing dataset")
    #parser.add_argument('--digit', default=5, type=int,
    #                    help="Digit to manipulate")
    #parser.add_argument('-w', '--weights', default=None,
    #                    help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # ---
    #  -CryTech Data: Change according to data
    #-----
    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # ---
    #  -CryTech makecaps accepts one parameter indicating the number of n_classes
    #  -It is 13 for character recognition case
    #-----
    # define model
    model = makeCaps(13)
    model.summary()
    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
