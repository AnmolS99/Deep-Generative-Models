from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from verification_net import VerificationNet
from stacked_mnist import DataMode, StackedMNISTData


class AutoEncoder(Model):

    def __init__(self, latent_dim) -> None:
        super(AutoEncoder, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(latent_dim, activation="relu")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(28**2, activation="sigmoid"),
            layers.Reshape((28, 28, 1)),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE,
                       default_batch_size=2048)

x_train, y_train = gen.get_full_data_set(training=True)
x_test, y_test = gen.get_full_data_set(training=False)

x_train = x_train[:, :, :, [0]]
x_test = x_test[:, :, :, [0]]

latent_dim = 4
autoencoder = AutoEncoder(latent_dim)
autoencoder.compile(optimizer='adam', loss=losses.BinaryCrossentropy())

autoencoder.fit(x_train,
                x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

ver_net = VerificationNet()
pred, acc = ver_net.check_predictability(decoded_imgs, y_test)
print("Pred: " + str(pred) + ", acc:" + str(acc))

z = np.random.randn(100, latent_dim)
decoded_imgs_random = autoencoder.decoder(z).numpy()
quality, _ = ver_net.check_predictability(decoded_imgs_random)
coverage = ver_net.check_class_coverage(decoded_imgs_random)
print("Quality: " + str(quality))
print("Coverage: " + str(coverage))

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display reconstruction
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(decoded_imgs_random[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()