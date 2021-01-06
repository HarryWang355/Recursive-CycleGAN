from abc import ABC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import numpy as np
import os
from PIL import Image, ImageOps


def downsample(filters, size, strides=2, use_normalization=True):
    """
    Implements downsampling (convolution + instance normalization + relu)
    :param strides: strides in the convolution layer
    :param filters: number of output filters
    :param size: kernel size
    :param use_normalization: a boolean for whether to apply instance normalization
    :return: a keras model
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = keras.Sequential()
    model.add(layers.Conv2D(filters, size, strides=strides, padding='same',
                            kernel_initializer=initializer, use_bias=False))
    if use_normalization:
        model.add(tfa.layers.InstanceNormalization(axis=-1, gamma_initializer=gamma_init))

    model.add(layers.ReLU())

    return model


def upsample(filters, size, strides=2):
    """
    Implements upsampling (transposed convolution + instance normalization + relu)
    :param filters: number of output filters
    :param size: kernel size
    :param strides: strides in the convolution layer
    :return: a keras model
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = keras.Sequential()
    model.add(layers.Conv2DTranspose(filters, size, strides=strides,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     use_bias=False))

    model.add(tfa.layers.InstanceNormalization(axis=-1, gamma_initializer=gamma_init))

    model.add(layers.ReLU())

    return model


def residual_layer(x0):
    """
    Implements a residual layer
    :param x0: input tensor
    :return: a tensor
    """
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    k = int(x0.shape[-1])

    # first layer
    x = keras.layers.Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x0)
    x = tfa.layers.InstanceNormalization(axis=-1, gamma_initializer=gamma_init)(x, training=True)
    x = keras.layers.Activation('relu')(x)

    # second layer
    x = keras.layers.Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x)
    x = tfa.layers.InstanceNormalization(axis=-1, gamma_initializer=gamma_init)(x, training=True)

    # merge
    x = tf.add(x, x0)
    return x


def Generator(num_channels=3):
    """
    Implements a generator
    :param num_channels: number of input and output channels
    :return: a generator model
    """
    inputs = layers.Input(shape=[256, 256, num_channels])

    # downsampling
    x = downsample(32, 7, strides=1, use_normalization=False)(inputs)
    x = downsample(64, 3)(x)
    x = downsample(128, 3)(x)

    # residual layers
    for _ in range(8):
        x = residual_layer(x)

    # upsampling
    x = upsample(64, 3)(x)
    x = upsample(32, 3)(x)

    # output layers
    x = keras.layers.Conv2D(num_channels, kernel_size=7, strides=1, padding='same')(x)
    x = keras.layers.Activation('tanh')(x)

    return keras.Model(inputs=inputs, outputs=x)


def Discriminator(num_channels=3):
    """
    Implements a discriminator
    :param num_channels: number of input and output channels
    :return: a discriminator model
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, num_channels], name='input_image')

    x = downsample(64, 4, use_normalization=False)(inp)  # (bs, 128, 128, 64)
    x = downsample(128, 4)(x)  # (bs, 64, 64, 128)
    x = downsample(256, 4)(x)  # (bs, 32, 32, 256)

    x = layers.ZeroPadding2D()(x)  # (bs, 34, 34, 256)
    x = layers.Conv2D(512, 4, strides=1,
                      kernel_initializer=initializer,
                      use_bias=False)(x)  # (bs, 31, 31, 512)

    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

    x = layers.LeakyReLU()(x)

    x = layers.ZeroPadding2D()(x)  # (bs, 33, 33, 512)

    x = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=x)


def discriminator_loss(real, generated):
    """
    Calculates discriminator loss
    :param real: real image
    :param generated: generated image
    :return: the loss
    """
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(real), real)

    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    """
    Calculates the generator loss
    :param generated: generated image
    :return:
    """
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(generated), generated)


def cycle_loss(real_image, cycled_image, LAMBDA):
    """
    Calculates the cycle consistency loss
    :param real_image: real image
    :param cycled_image: reconstructed image
    :param LAMBDA: the weight
    :return: the cycle consistency loss
    """
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image, LAMBDA):
    """
    Calculates the identity loss
    :param real_image: real image
    :param same_image: same image
    :param LAMBDA: the weight
    :return: the identity loss
    """
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


class RecursiveCycleGAN(keras.Model, ABC):
    """
    Implements the Recursive CycleGAN class
    """

    def __init__(self, num_cycle=2, channel_num=3, lambda_cycle=10, alpha=1):
        super(RecursiveCycleGAN, self).__init__()
        self.G_X2Y = Generator(channel_num)  # generator from X to Y
        self.G_Y2X = Generator(channel_num)  # generator from Y to X
        self.D_Y = Discriminator(channel_num)  # discriminator of Y
        self.D_X = Discriminator(channel_num)  # discriminator of X
        self.lambda_cycle = lambda_cycle  # weight on cycle consistency loss
        self.num_cycle = num_cycle  # number of cycles
        self.alpha = alpha  # decay rate on the loss of cycles

        # initialize optimizers
        self.G_X2Y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.G_Y2X_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.D_X_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.D_Y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # define loss functions
        self.gen_loss_fn = generator_loss
        self.disc_loss_fn = discriminator_loss
        self.cycle_loss_fn = cycle_loss
        self.identity_loss_fn = identity_loss

    def train_step(self, batch_data):
        real_X, real_Y = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # initialize total losses
            final_G_X2Y_loss = 0
            final_G_Y2X_loss = 0
            final_D_X_loss = 0
            final_D_Y_loss = 0

            input_X = real_X
            input_Y = real_Y

            for i in range(self.num_cycle):
                # X to Y back to X
                fake_Y = self.G_X2Y(input_X, training=True)
                cycled_X = self.G_Y2X(fake_Y, training=True)

                # Y to X back to Y
                fake_X = self.G_Y2X(input_Y, training=True)
                cycled_Y = self.G_X2Y(fake_X, training=True)

                # generating itself
                same_Y = self.G_X2Y(input_Y, training=True)
                same_X = self.G_Y2X(input_X, training=True)

                # discriminator used to check, inputing real images
                disc_real_Y = self.D_Y(input_Y, training=True)
                disc_real_X = self.D_X(input_X, training=True)

                # discriminator used to check, inputing fake images
                disc_fake_Y = self.D_Y(fake_Y, training=True)
                disc_fake_X = self.D_X(fake_X, training=True)

                # evaluates generator loss
                G_X2Y_loss = self.gen_loss_fn(disc_fake_Y)
                G_Y2X_loss = self.gen_loss_fn(disc_fake_X)

                # evaluates total cycle consistency loss
                total_cycle_loss = self.cycle_loss_fn(input_Y, cycled_Y, self.lambda_cycle) + self.cycle_loss_fn(
                    input_X, cycled_X, self.lambda_cycle)

                # evaluates total generator loss in this cycle
                total_G_X2Y_loss = G_X2Y_loss + total_cycle_loss + \
                                   self.identity_loss_fn(input_Y, same_Y, self.lambda_cycle)
                total_G_Y2X_loss = G_Y2X_loss + total_cycle_loss + \
                                   self.identity_loss_fn(input_X, same_X, self.lambda_cycle)

                # evaluates discriminator loss in this cycle
                D_Y_loss = self.disc_loss_fn(disc_real_Y, disc_fake_Y)
                D_X_loss = self.disc_loss_fn(disc_real_X, disc_fake_X)

                # update total losses
                final_G_X2Y_loss += (self.alpha ** i) * total_G_X2Y_loss
                final_G_Y2X_loss += (self.alpha ** i) * total_G_Y2X_loss
                final_D_X_loss += (self.alpha ** i) * D_X_loss
                final_D_Y_loss += (self.alpha ** i) * D_Y_loss

                # update input values for the next round
                input_X = cycled_X
                input_Y = cycled_Y

        # Calculate the gradients for generator and discriminator
        G_X2Y_gradients = tape.gradient(final_G_X2Y_loss, self.G_X2Y.trainable_variables)
        G_Y2X_gradients = tape.gradient(final_G_Y2X_loss, self.G_Y2X.trainable_variables)

        D_X_gradients = tape.gradient(final_D_X_loss, self.D_X.trainable_variables)
        D_Y_gradients = tape.gradient(final_D_Y_loss, self.D_Y.trainable_variables)

        # Apply the gradients to the optimizer
        self.G_X2Y_optimizer.apply_gradients(zip(G_X2Y_gradients,
                                                 self.G_X2Y.trainable_variables))

        self.G_Y2X_optimizer.apply_gradients(zip(G_Y2X_gradients,
                                                 self.G_Y2X.trainable_variables))

        self.D_X_optimizer.apply_gradients(zip(D_X_gradients,
                                               self.D_X.trainable_variables))

        self.D_Y_optimizer.apply_gradients(zip(D_Y_gradients,
                                               self.D_Y.trainable_variables))

        return {
            "G_X2Y_loss": final_G_X2Y_loss,
            "G_Y2X_loss": final_G_Y2X_loss,
            "D_X_loss": final_D_X_loss,
            "D_Y_loss": final_D_Y_loss
        }


if __name__ == '__main__':

    model = RecursiveCycleGAN()
