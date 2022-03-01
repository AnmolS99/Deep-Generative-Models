import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.models import Model

tfk = tf.keras
tfkl = tfk.layers
tfd = tfp.distributions
tfpl = tfp.layers


class VariationalAutoEncoder(Model):

    def __init__(self,
                 latent_dim,
                 filename="./models/variational_autoencoder_weights") -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.filename = filename

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim),
                                                scale=1),
                                     reinterpreted_batch_ndims=1)

        self.encoder = tfk.Sequential([
            tfkl.Input(shape=(28, 28, 1)),
            tfkl.Flatten(),
            tfkl.Dense(784, activation=tf.nn.leaky_relu),
            tfkl.Dense(512, activation=tf.nn.leaky_relu),
            tfkl.Dense(128, activation=tf.nn.leaky_relu),
            tfkl.Dense(32, activation=tf.nn.leaky_relu),
            tfkl.Dense(16, activation=tf.nn.leaky_relu),
            tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim),
                       activation=None),
            tfpl.MultivariateNormalTriL(
                latent_dim,
                activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior)),
        ])

        self.decoder = tfk.Sequential([
            tfkl.Dense(16, activation=tf.nn.leaky_relu),
            tfkl.Dense(32, activation=tf.nn.leaky_relu),
            tfkl.Dense(128, activation=tf.nn.leaky_relu),
            tfkl.Dense(512, activation=tf.nn.leaky_relu),
            tfkl.Dense(784),
            tfpl.IndependentBernoulli((28, 28, 1), tfd.Bernoulli.logits),
        ])

        # Compiling the model
        negloglik = lambda x, rv_x: -rv_x.log_prob(x)
        self.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                     loss=negloglik)

    def load_var_autoencoder_weights(self):
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

        self.done_training = self.load_var_autoencoder_weights()

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

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
