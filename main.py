import tensorflow  as tf
from GAN import GAN, train_GAN, generate_image
from settings import epochs, latent_dim

trained = True

if trained:
    model = GAN()
    _ = model(tf.zeros((1, latent_dim)))
    model.load_weights('gan_weights.h5')
else:
    model, history = train_GAN(epochs)

generate_image(model, latent_dim, num_images=10)