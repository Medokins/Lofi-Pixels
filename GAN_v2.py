import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, BatchNormalization, Conv2DTranspose, Conv2D
from keras.models import Model

params = {
    "data_path": os.path.join("data", "selected"),  # path to directory with images
    "latent_dim": 100,
    "im_height": 512,
    "im_width": 512,
    "batch_size": 64,
    "epochs": 1000,
    "d_optimizer": tf.keras.optimizers.Adamax(),        # 
    "g_optimizer": tf.keras.optimizers.Adamax(),        #  optimizers for compile method
    "loss_n": tf.keras.losses.BinaryCrossentropy(),     #

    "trained": False,
    "num_images": 2,
}

class GAN_v2(tf.keras.Model):
    def __init__(self):  
        if len(tf.config.list_physical_devices('GPU')):
            print("Using GPU")
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_visible_devices(gpu_devices[0], 'GPU')

        super(GAN_v2, self).__init__()
        self.data_path = params["data_path"]
        self.latent_dim = params["latent_dim"]
        self.im_height = params["im_height"]
        self.im_width = params["im_width"]
        self.batch_size = params["batch_size"]
        self.normalized_data = self.get_data()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.compile(params["d_optimizer"], params["g_optimizer"], params["loss_n"])

    # method to be load trained model
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal((tf.shape(inputs)[0], self.latent_dim))
            fake_images = self.generator(noise, training=training)
            disc_fake = self.discriminator(fake_images, training=training)
            return disc_fake
        else:
            return None

    def get_data(self):
        data = tf.keras.preprocessing.image_dataset_from_directory(self.data_path, label_mode=None, image_size=(self.im_height, self.im_width), batch_size=self.batch_size)
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        return data.map(lambda x: normalization_layer(x))

    def build_generator(self):
        input_layer = Input(shape=(self.latent_dim, ))

        dense_layer = Dense(8*8*1024, activation='relu')(input_layer)
        reshape_layer = Reshape((8, 8, 1024))(dense_layer)

        upsample_layer1 = Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', activation='relu')(reshape_layer)
        bn_layer1 = BatchNormalization()(upsample_layer1)
        upsample_layer2 = Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', activation='relu')(bn_layer1)
        bn_layer2 = BatchNormalization()(upsample_layer2)
        upsample_layer3 = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu')(bn_layer2)
        bn_layer3 = BatchNormalization()(upsample_layer3)
        upsample_layer4 = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(bn_layer3)
        bn_layer4 = BatchNormalization()(upsample_layer4)
        upsample_layer5 = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu')(bn_layer4)
        bn_layer5 = BatchNormalization()(upsample_layer5)

        output_layer = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(bn_layer5)

        generator = Model(input_layer, output_layer, name='generator')
        return generator

    def build_discriminator(self):
        img_shape = (self.im_height, self.im_width, 3)
        input_layer = Input(shape=img_shape)

        conv_layer1 = Conv2D(32, kernel_size=3, strides=2, padding='same')(input_layer)
        leaky_relu1 = LeakyReLU(alpha=0.2)(conv_layer1)
        dropout_layer1 = Dropout(0.25)(leaky_relu1)

        conv_layer2 = Conv2D(64, kernel_size=3, strides=2, padding='same')(dropout_layer1)
        bn_layer2 = BatchNormalization()(conv_layer2)
        leaky_relu2 = LeakyReLU(alpha=0.2)(bn_layer2)
        dropout_layer2 = Dropout(0.25)(leaky_relu2)

        conv_layer3 = Conv2D(128, kernel_size=3, strides=2, padding='same')(dropout_layer2)
        bn_layer3 = BatchNormalization()(conv_layer3)
        leaky_relu3 = LeakyReLU(alpha=0.2)(bn_layer3)
        dropout_layer3 = Dropout(0.25)(leaky_relu3)

        conv_layer4 = Conv2D(256, kernel_size=3, strides=2, padding='same')(dropout_layer3)
        bn_layer4 = BatchNormalization()(conv_layer4)
        leaky_relu4 = LeakyReLU(alpha=0.2)(bn_layer4)
        dropout_layer4 = Dropout(0.25)(leaky_relu4)

        flatten_layer = Flatten()(dropout_layer4)

        output_layer = Dense(1, activation='sigmoid')(flatten_layer)
        discriminator = Model(input_layer, output_layer, name='discriminator')
        discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

        return discriminator
    
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN_v2, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.discriminator_loss = tf.keras.metrics.Mean(name="discriminator_loss")
        self.generator_loss = tf.keras.metrics.Mean(name="generator_loss")

    def train(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for i, images in enumerate(self.normalized_data):
                batch_size = images.shape[0]
                noise = tf.random.normal((batch_size, self.latent_dim))

                with tf.GradientTape() as tape:
                    generated_images = self.generator(noise, training=True)

                    real_outputs = self.discriminator(images, training=True)
                    fake_outputs = self.discriminator(generated_images, training=True)

                    d_real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_outputs), real_outputs)
                    d_fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_outputs), fake_outputs)
                    d_loss = d_real_loss + d_fake_loss

                d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
                self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

                with tf.GradientTape() as tape:
                    generated_images = self.generator(noise, training=True)
                    fake_outputs = self.discriminator(generated_images, training=True)

                    g_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_outputs), fake_outputs)

                g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
                self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

                if (i+1) % 10 == 0:
                    print(f"Batch {i+1}/{len(self.normalized_data)} - d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}")

        name = params["data_path"].split("\\")[1]
        self.save_weights(f'Gan_v2_{params["epochs"]}_{params["batch_size"]}_{name}.h5')
        

def generate_images(model, latent_dim, num_images):
    noise = tf.random.normal([num_images, latent_dim])
    generated_images = model.generator(noise, training=False)
    generated_images = generated_images.numpy()

    _, axs = plt.subplots(1, num_images, figsize=(20, 20))
    for i in range(num_images):
        axs[i].imshow(generated_images[i])
        axs[i].axis('off')
    plt.show()

def run_GAN(trained=False):
    model = GAN_v2()
    if trained:
        name = params["data_path"].split("\\")[1]
        model(tf.zeros((1, params["latent_dim"])))
        model.load_weights(f'Gan_v2_{params["epochs"]}_{params["batch_size"]}_{name}.h5')
    else:
        model.train(params["epochs"])
    generate_images(model, params["latent_dim"], params["num_images"])

run_GAN(trained=params["trained"])