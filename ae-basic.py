from matplotlib import pyplot as plt
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

# Sending the images through the AE to get reconstructed images
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# Using VerNet to get predictability and accuracy
ver_net = VerificationNet()
pred, acc = ver_net.check_predictability(decoded_imgs, y_test)
print("Pred: " + str(pred) + ", acc:" + str(acc))

# Showing the original images and reconstructed images
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
plt.suptitle("" + str(n) + " images reconstructed", fontsize="x-large")

# Save figure
plt.savefig("./results/ae-basic")

# Show image
plt.show()