import os

import numpy as np

from vae import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150  # init is 20

SPECTROGRAMS_PATH = "/Users/johnrupsch/FirstAutoEncoder/datasets/fsdd/spectrograms"

def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (3000, 256, 64, 1)
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train = load_fsdd(SPECTROGRAMS_PATH)
    autoencoder = train(x_train[:500], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
    """autoencoder2 = VAE.load("model")"""
    """autoencoder2.summary()"""
