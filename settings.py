import os

# NOTE, this is used only for GAN.py, GAN_v2.py (better working model) has its parameters stored inside params dict
latent_dim = 100
g_resolution = 2
d_resolution = 128

im_height, im_width = 512, 512
batch_size = 64
epochs = 1000

path = os.path.join("data", "selected")