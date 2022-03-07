import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


class AutoEncoder(Model):

    def __init__(self,
                 latent_dim,
                 filename="./models/autoencoder_weights") -> None:
        super(AutoEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.filename = filename

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, (3, 3),
                          activation=tf.nn.leaky_relu,
                          padding='same',
                          strides=2),
            layers.Conv2D(32, (3, 3),
                          activation=tf.nn.leaky_relu,
                          padding='same',
                          strides=2),
            layers.Flatten(),
            layers.Dense(256, activation=tf.nn.leaky_relu),
            layers.Dense(128, activation=tf.nn.leaky_relu),
            layers.Dense(64, activation=tf.nn.leaky_relu),
            layers.Dense(32, activation=tf.nn.leaky_relu),
            layers.Dense(16, activation=tf.nn.leaky_relu),
            layers.Dense(latent_dim, activation=tf.nn.leaky_relu)
        ])

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim, )),
            layers.Dense(16, activation=tf.nn.leaky_relu),
            layers.Dense(32, activation=tf.nn.leaky_relu),
            layers.Dense(64, activation=tf.nn.leaky_relu),
            layers.Dense(128, activation=tf.nn.leaky_relu),
            layers.Dense(256, activation=tf.nn.leaky_relu),
            layers.Dense(7 * 7 * 16, activation=tf.nn.leaky_relu),
            layers.Reshape((7, 7, 16)),
            layers.Conv2DTranspose(32,
                                   kernel_size=3,
                                   strides=2,
                                   activation=tf.nn.leaky_relu,
                                   padding='same'),
            layers.Conv2DTranspose(64,
                                   kernel_size=3,
                                   strides=2,
                                   activation=tf.nn.leaky_relu,
                                   padding='same'),
            layers.Conv2D(1,
                          kernel_size=(3, 3),
                          activation='sigmoid',
                          padding='same')
        ])

        # Compiling the autoencoder
        self.compile(optimizer='adam', loss=losses.BinaryCrossentropy())

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def load_autoencoder_weights(self):
        # noinspection PyBroadException
        try:
            self.load_weights(filepath=self.filename)
            print(f"Read model from file, so I do not retrain")
            done_training = True

        except:
            print(
                f"Could not read weights for verification_net from file. Must retrain..."
            )
            done_training = False

        return done_training

    def train(self,
              x=None,
              y=None,
              batch_size=None,
              epochs=1,
              shuffle=True,
              validation_data=None,
              verbose=True,
              save_weights=False):

        self.done_training = self.load_autoencoder_weights()

        if save_weights or not self.done_training:

            self.fit(x=x,
                     y=y,
                     batch_size=batch_size,
                     epochs=epochs,
                     shuffle=shuffle,
                     validation_data=validation_data,
                     verbose=verbose)

            # Save weights and leave
            self.save_weights(filepath=self.filename)
            self.done_training = True

        return self.done_training
