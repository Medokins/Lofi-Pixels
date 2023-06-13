import os
import tensorflow as tf

params = {
    "data_path": os.path.join("data", "converted"),  # path to directory with images on which GAN is trained
    "latent_dim": 100,
    "im_height": 512,
    "im_width": 512,
    "batch_size": 64,
    "epochs": 256,
    "d_optimizer": tf.keras.optimizers.Adamax(),        # 
    "g_optimizer": tf.keras.optimizers.Adamax(),        #  optimizers for compile method
    "loss_n": tf.keras.losses.BinaryCrossentropy(),     #

    "trained": False, # whether or not the model was trained
    "num_images": 5,  # num of images to generate 
}