import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from stacked_mnist import DataMode, StackedMNISTData
from variational_autoencoder import VariationalAutoEncoder
from verification_net import VerificationNet

latent_dim = 3
vae = VariationalAutoEncoder(
    latent_dim, filename="./models/var_autoencoder_weights_mono_missing")

# Creating generators using standard MNIST
gen_missing = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING,
                               default_batch_size=2048)

gen_complete = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE,
                                default_batch_size=2048)

# Getting training and test data
x_train, y_train = gen_missing.get_full_data_set(training=True)
x_test, y_test = gen_complete.get_full_data_set(training=False)

# Reshaping
x_train = x_train[:, :, :, [0]]
x_test = x_test[:, :, :, [0]]

vae.train(x=x_train,
          y=x_train,
          batch_size=512,
          epochs=100,
          validation_data=(x_test, x_test),
          save_weights=False)

k = 24
n = 1000
x_test_set = x_test[:1000]
z = vae.prior.sample(n)
decoded_z_imgs = vae.decoder(z).mode().numpy()
bin_cross = tf.keras.losses.BinaryCrossentropy()
probs = []
x_num = 0
for x in x_test_set:
    print("On x_test_set case: " + str(x_num))
    x_num += 1

    x_prob = []
    for i in range(n):
        neglogprob = bin_cross(x, decoded_z_imgs[i])
        prob = np.exp(-neglogprob)
        x_prob.append(prob)
    x_prob = np.array(x_prob)

    # Taking the mean
    probs.append(np.mean(x_prob))

probs = np.array(probs)

# Index of the k-lowest values
idx = np.argpartition(probs, k)
lowest_prob_x = x_test_set[idx[:k]]

# Showing the original images and reconstructed images
plt.figure(figsize=(20, 6))
for i in range(k):
    # display original
    ax = plt.subplot(3, k // 3, i + 1)
    plt.imshow(lowest_prob_x[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.suptitle("" + str(k) + " most anomolous images ", fontsize="x-large")

# Save figure
#plt.savefig("./results/vae-anom")

# Show image
plt.show()