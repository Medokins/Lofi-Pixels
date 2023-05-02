import tensorflow  as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, LeakyReLU, Dropout, Flatten, Activation
from settings import *

class GAN(tf.keras.Model):
    def __init__(self):
        super(GAN, self).__init__()
        self.normalized_data = self.get_data()
        self.generator = self.get_generator()
        self.discriminator = self.get_discriminator()
        self.latent_dim = latent_dim

        if len(tf.config.list_physical_devices('GPU')):
            print("Using GPU")
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_visible_devices(gpu_devices[0], 'GPU')

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal((tf.shape(inputs)[0], self.latent_dim))
            fake_images = self.generator(noise, training=training)
            disc_fake = self.discriminator(fake_images, training=training)
            return disc_fake
        else:
            return None


    def get_data(self):
        data = tf.keras.preprocessing.image_dataset_from_directory(path, label_mode=None, image_size=(im_height, im_width), batch_size=batch_size)
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        return data.map(lambda x: normalization_layer(x))

    def get_generator(self, verbose=False):
        generator = Sequential()
        generator.add(Dense(4*4*256, activation='relu', input_dim=latent_dim))
        generator.add(Reshape((4, 4, 256)))
        generator.add(UpSampling2D(size=(g_resolution, g_resolution)))
        generator.add(Conv2D(256, kernel_size=3, padding='same'))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Activation('relu'))
        generator.add(UpSampling2D(size=(g_resolution, g_resolution)))
        generator.add(Conv2D(256, kernel_size=3, padding='same'))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Activation('relu'))
        generator.add(UpSampling2D(size=(g_resolution, g_resolution)))
        generator.add(Conv2D(256, kernel_size=3, padding='same'))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Activation('relu'))
        generator.add(UpSampling2D(size=(g_resolution, g_resolution)))
        generator.add(Conv2D(128, kernel_size=3, padding='same'))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Activation('relu'))
        generator.add(UpSampling2D(size=(g_resolution, g_resolution)))
        generator.add(Conv2D(3, kernel_size=3, padding='same'))
        generator.add(Activation('tanh'))
        if verbose:
            print(generator.summary())
        return generator

    def get_discriminator(self, verbose=False):
        discriminator = Sequential()
        discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(d_resolution, d_resolution, 3), padding='same'))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation='sigmoid'))
        if verbose:
            print(discriminator.summary())
        return discriminator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.discriminator_loss = tf.keras.metrics.Mean(name="discriminator_loss")
        self.generator_loss = tf.keras.metrics.Mean(name="generator_loss")

    @property
    def metrics(self):
        return [self.discriminator_loss, self.generator_loss]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        seed = tf.random.normal(shape=(batch_size, self.latent_dim))

        generated_images = self.generator(seed)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels += 0.02137 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            discriminator_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        seed = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(seed))
            generator_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.discriminator_loss.update_state(discriminator_loss)
        self.generator_loss.update_state(generator_loss)
        return {"discriminator_loss": self.discriminator_loss.result(), "generator_loss": self.generator_loss.result()}
    

def train_GAN(n_epochs):
    discriminator_opt = tf.keras.optimizers.Adamax()
    generator_opt = tf.keras.optimizers.Adamax()
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    model = GAN()
    model.compile(d_optimizer=discriminator_opt, g_optimizer=generator_opt, loss_fn=loss_fn)
    history = model.fit(model.normalized_data, epochs=n_epochs)
    model.save_weights('gan_weights.h5')

    return model, history


def generate_image(model, latent_dim, num_images):
    noise = tf.random.normal([num_images, latent_dim])
    generator = model.generator
    generated_images = generator(noise, training=False)

    _, axs = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 15))
    for i in range(num_images):
        axs[i].imshow(generated_images[i] * 0.5 + 0.5)
        axs[i].axis('off')
    plt.show()
