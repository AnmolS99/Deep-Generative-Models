from matplotlib import pyplot as plt
import numpy as np
from autoencoder import AutoEncoder
from stacked_mnist import DataMode, StackedMNISTData
from verification_net import VerificationNet

# Creating a generator using standard MNIST
gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE,
                       default_batch_size=2048)

# Getting training and test data
x_train, y_train = gen.get_full_data_set(training=True)
x_test, y_test = gen.get_full_data_set(training=False)

# Reshaping
x_train = x_train[:, :, :, [0]]
x_test = x_test[:, :, :, [0]]

# Creating AutoEncoder
latent_dim = 4
autoencoder = AutoEncoder(latent_dim)

# Training the AE
autoencoder.train(x_train,
                  x_train,
                  epochs=20,
                  shuffle=True,
                  validation_data=(x_test, x_test),
                  verbose=True,
                  save_weights=False)

# Creating VerNet
ver_net = VerificationNet()

# Creating random veactors to push through the decoder
z = np.random.randn(100, latent_dim)
decoded_imgs_random = autoencoder.decoder(z).numpy()

# Printing out quality and coverage
quality, _ = ver_net.check_predictability(decoded_imgs_random)
coverage = ver_net.check_class_coverage(decoded_imgs_random)
print("Quality: " + str(quality))
print("Coverage: " + str(coverage))

# Showing the decoded images
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