from tensorflow.keras import layers, models
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from layer import *
from matplotlib import pyplot as plt

def makeCaps(int numcaps):
    # Building a Capsule Network as specified by Sara Sabour
    # Start with a Convolution Layer with 256 9*9 filters
    # output would be  20*20*256 images

    #creating input layer
    # ---
    #  -CryTech Size : 28, 28, 1
    #-----
    x = tf.keras.Input(shape=(28,28,1))
    #creating conv layer as mentioned before
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu')(x)
    #primary capsule which involves
    #we need 32 capsules each capsule has 8 dimensional array of size 6*6
    #Conv layer is used with no of filters as 32*8 = 256
    #reshape the array from 6*6*256 as (6*6*32)*8
    output = layers.Conv2D(filters=256, kernel_size=9, strides=2, padding='valid')(conv1)
    # ---
    #  -CryTech Primary Capsule Size : 6*6*32 capsules of 8Dimension
    #-----
    outputs = layers.Reshape(target_shape=[1152, 8], name='primarycap_reshape')(output)
    #Apply Squash
    outputs = layers.Lambda(squash, name='primarycap_squash')(outputs)
    #Create a Capsule Layer
    # ---
    #  -CryTech Primary Capsule Size : 13 capsules(representing 13 classes) of 10Dimension
    #   :param number of routings is set to 3
    #-----
    digitcaps = CapsuleLayer(num_capsule=numcaps, dim_capsule=16, routings=3,name='digitcaps')(outputs)
    out_caps = Length(name='capsnet')(digitcaps)


    y = layers.Input(shape=(numcaps,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*numcaps))
    decoder.add(layers.Dense(1024, activation='relu'))
    # ---
    #  -CryTech Input Shape = 28 * 28 = 784
    #-----
    decoder.add(layers.Dense(784, activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=(28,28,1), name='out_recon'))


    M = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    return M


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()





def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model
