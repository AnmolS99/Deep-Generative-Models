import matplotlib.pyplot as plt

from stacked_mnist import DataMode, StackedMNISTData
from variational_autoencoder import VariationalAutoEncoder
from verification_net import VerificationNet

latent_dim = 3
vae = VariationalAutoEncoder(latent_dim)

gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE,
                       default_batch_size=2048)
# Getting training and test data
x_train, y_train = gen.get_full_data_set(training=True)
x_test, y_test = gen.get_full_data_set(training=False)

# Reshaping
x_train = x_train[:, :, :, [0]]
x_test = x_test[:, :, :, [0]]

vae.train(x=x_train,
          y=x_train,
          batch_size=512,
          epochs=100,
          validation_data=(x_test, x_test),
          save_weights=False)

# Sending the images through the AE to get reconstructed images
z_sample = vae.prior.sample(10)
decoded_imgs_mode = vae.decoder(z_sample).mode().numpy()
decoded_imgs_mean = vae.decoder(z_sample).mean().numpy()

# Using VerNet to get predictability and accuracy
ver_net = VerificationNet()
pred, acc = ver_net.check_predictability(decoded_imgs_mode)
coverage = ver_net.check_class_coverage(decoded_imgs_mode)
print("For mode - Pred: " + str(pred) + ", coverage: " + str(coverage))

pred, acc = ver_net.check_predictability(decoded_imgs_mean)
coverage = ver_net.check_class_coverage(decoded_imgs_mean)
print("For mode - Pred: " + str(pred) + ", coverage: " + str(coverage))

# Showing the original images and reconstructed images
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # display generated images by taking mode of output
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(decoded_imgs_mode[i])
    plt.title("gen. mode")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display generated images by taking mean of output
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_mean[i])
    plt.title("gen. mean")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.suptitle("" + str(n) + " images generated", fontsize="x-large")

# Save figure
plt.savefig("./results/vae-gen")

# Show image
plt.show()