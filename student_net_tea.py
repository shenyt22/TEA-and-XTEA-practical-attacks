import tea

import numpy as np

from util import select_tar_bits
from pickle import dump

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

bs = 5000
wdir = './freshly_trained_nets/'


def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return res


def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True)
  return res


#make residual tower of convolutional blocks
def make_resnet(num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=32, ks=3, depth=5, reg_param=0.0001, final_activation='sigmoid'):
    inp = Input(shape=(num_blocks * word_size ,))
    rs = Reshape((num_blocks, word_size))(inp)
    perm = Permute((2,1))(rs)
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return model


def train_tea_distinguisher(num_epochs, nr_range, depth=1, diff=(0x80402010, 0), folder='./', selected_bits=[38, 37, 33, 32, 1, 0], net_suffix="_1"):
    #create the network
    net = make_resnet(depth=depth, reg_param=10**-5, word_size=len(selected_bits))
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    #generate training and validation data
    num_rounds = 0
    for _ in nr_range:
      num_rounds += 1
    X, Y = tea.make_train_data(10**7, nr_range, diff=diff)
    X = select_tar_bits(X, selected_bits)
    X_eval, Y_eval = tea.make_train_data(10**6, nr_range, diff=diff)
    X_eval = select_tar_bits(X_eval, selected_bits)
    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5')
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    #train and evaluate
    h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), shuffle=True, callbacks=[lr, check])
    dump(h.history, open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))

    # save model
    net.save(folder + str(num_rounds) + '_distinguisher' + net_suffix + '.h5')

    return (net, h)

if __name__ == '__main__':
  net = make_resnet(depth=1, reg_param=10**(-5))
  net.summary()