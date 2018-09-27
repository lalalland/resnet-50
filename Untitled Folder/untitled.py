from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose,GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras.optimizers import RMSprop

import tensorflow as tf

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


    
def encoder(x, filters=64, n_block=3, kernel_size=(3, 3), activation='relu'):
    skip = []
    for i in range(n_block):
        print('*********************encoder {}******************'.format(i+1))
        print('input {}'.format(x.shape))
        x = Conv2D(filters * 2**i, kernel_size, activation=None, padding='same')(x)
        print('conv no act {}'.format(x.shape))
#         x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = residual_block(x,filters * 2**i)
        print('conv 1 {}'.format(x.shape))
#         x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = residual_block(x,filters * 2**i,True)
        print('conv 2 {}'.format(x.shape))
        x = squeeze_excite_block(x, ratio=16)
        print('squeeze{}'.format(x.shape))
        skip.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
#         x = Dropout(DropoutRatio/2)(x)
        print('Max {}'.format(x.shape))
        print('**************************************************************************')
    return x, skip


def bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
               kernel_size=(3, 3), activation='relu'):
    dilated_layers = []
    print('*********************************middle*********************************')
    print('input {}'.format(x.shape))
    if mode == 'cascade':  # used in the competition
        print('cascade dilatation')
        for i in range(depth):
            
            x = Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
           
            dilated_layers.append(x)
            print('conv dilatation {} {}'.format(i,x.shape))
        return add(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        print('parallel dilatation')
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            )
            print('conv dilatation {} {}'.format(i,Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x).shape))
        return add(dilated_layers)


def decoder(x, skip, filters, n_block=3, kernel_size=(3, 3), activation='relu'):
    
    
    for i in reversed(range(n_block)):
        print('*********************************decoder{}*********************************'.format(i+1))
        print('input {}'.format(x.shape))
#         x = UpSampling2D(size=(2, 2))(x)
        print('upsampling {}'.format(x.shape))
        print('filters {}'.format(filters * 2**i))
        print('kernel {}'.format(kernel_size))
        if i == 3 or i == 1:
            x = Conv2DTranspose(filters * 2**i, kernel_size, strides=(2, 2), padding="same")(x)
# #             x = Conv2D(filters * 2**i, kernel_size, activation=None,strides=(2, 2), padding='valid')(x)
# #             x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
#             x = Conv2DTranspose(filters * 2**i, kernel_size, strides=(2, 2), padding="same")(x)
        else:
            x = Conv2DTranspose(filters * 2**i, kernel_size, strides=(2, 2), padding="valid")(x)
#         print('conv 1 {}'.format(x.shape))
        print('skip {}'.format(skip[i].shape))
        print('conv T {}'.format(x))
        x = concatenate([skip[i], x])
        print('concat {}'.format(x.shape))
        x = Conv2D(filters * 2**i, kernel_size, activation=None, padding='same')(x)
        print('conv1 no activation {}'.format(x.shape))
#         x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = residual_block(x,filters * 2**i)
        print('conv2 {}'.format(x.shape))
#         x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = residual_block(x,filters * 2**i, True)
        print('conv3 {}'.format(x.shape))
        x = squeeze_excite_block(x, ratio=16)
        print('squeeze{}'.format(x.shape))
    return x


def get_dilated_unet(
        input_shape=(1920, 1280, 3),
        mode='cascade',
        filters=44,
        n_block=4,
        lr=0.0001,
        loss='bce_dice_loss',
        n_class=1
):
    inputs = Input(input_shape)
    
    enc, skip = encoder(inputs, filters, n_block)
    bottle = bottleneck(enc, filters_bottleneck=filters * 2**n_block, mode=mode)
    dec = decoder(bottle, skip, filters, n_block)
    classify = Conv2D(n_class, (1, 1), activation='sigmoid')(dec)
    print('last {}'.format(classify.shape))
    model = Model(inputs=inputs, outputs=classify)
#     model.compile(optimizer=RMSprop(lr), loss=loss, metrics=[dice_coef])

    return model