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
