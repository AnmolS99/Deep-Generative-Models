from matplotlib import pyplot as plt
import numpy as np
from autoencoder import AutoEncoder
from stacked_mnist import DataMode, StackedMNISTData

# Creating generators using standard MNIST
gen_missing = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING,
                               default_batch_size=2048)

gen_complete = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE,
                                default_batch_size=2048)

# Getting training (missing the number 8) and test data (includes all numbers)
x_train, y_train = gen_missing.get_full_data_set(training=True)
x_test, y_test = gen_complete.get_full_data_set(training=False)

# Reshaping
x_train = x_train[:, :, :, [0]]
x_test = x_test[:, :, :, [0]]

# Creating AutoEncoder
latent_dim = 4
autoencoder = AutoEncoder(latent_dim,
                          filename="./models/autoencoder_weights_mono_missing")

# Training the AE
autoencoder.train(x_train,
                  x_train,
                  epochs=20,
                  shuffle=True,
                  validation_data=(x_test, x_test),
                  verbose=True,
                  save_weights=False)

k = 10
losses = []
for x in x_test:
    x = x.reshape(1, 28, 28, 1)
    loss = autoencoder.evaluate(x, x)
    losses.append(loss)

losses = np.array(losses)
ind = np.argpartition(losses, -k)[-k:]
top_losses_x = x_test[ind]

# Showing the decoded images
plt.figure(figsize=(20, 4))
for i in range(k):
    # display image
    ax = plt.subplot(2, k, i + 1)
    plt.imshow(top_losses_x[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.suptitle("Top " + str(k) + " highest loss images", fontsize="x-large")

# Save figure
plt.savefig("./results/top-k-losses-ae-anom")

# Show image
plt.show()
