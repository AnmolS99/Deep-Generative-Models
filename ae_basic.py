from matplotlib import pyplot as plt
import numpy as np
from autoencoder import AutoEncoder
from stacked_mnist import DataMode, StackedMNISTData
from verification_net import VerificationNet


class AEBasic:
    """
    Autoencoder basic task: Reconstructing images
    """

    def __init__(self,
                 latent_dim=4,
                 three_colors=False,
                 save_weigths=False,
                 save_image=False) -> None:
        self.autoencoder = AutoEncoder(latent_dim)
        self.three_colors = three_colors
        self.save_weigths = save_weigths
        self.save_image = save_image
        self.gen = self.get_generator(self.three_colors)
        self.ver_net = VerificationNet()

    def get_generator(self, three_colors):
        """
        Returning the appropriate generator
        """
        # Returning a generator that uses standard MNIST
        if three_colors:
            return StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE,
                                    default_batch_size=2048)
        # Returning a generator that uses stacked MNIST
        else:
            return StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE,
                                    default_batch_size=2048)

    def get_train_test(self, gen):
        """
        Getting the train and test data
        """
        x_train, y_train = gen.get_full_data_set(training=True)
        x_test, y_test = gen.get_full_data_set(training=False)
        return x_train, y_train, x_test, y_test

    def train_autoencoder(self):
        """
        Training the autoencoder on single-channel images
        """
        x_train, y_train, x_test, y_test = self.get_train_test(self.gen)

        # Reshaping
        x_train = x_train[:, :, :, [0]]
        x_test = x_test[:, :, :, [0]]

        # Training the AE
        self.autoencoder.train(x_train,
                               x_train,
                               batch_size=64,
                               epochs=20,
                               shuffle=True,
                               validation_data=(x_test, x_test),
                               verbose=True,
                               save_weights=self.save_weigths)

    def run(self):
        """
        Reconstructing images and displaying the results
        """
        # Training the autoencoder
        self.train_autoencoder()

        x_train, y_train, x_test, y_test = self.get_train_test(self.gen)

        # If we have multi-color images
        if self.three_colors:

            reconstructed = []

            # For each color-channel
            for i in range(3):
                # Getting the specific color channel
                x_test_channel = x_test[:, :, :, [i]]

                # Sending the images through the AE to get reconstructed images
                encoded_imgs = self.autoencoder.encoder(x_test_channel).numpy()
                decoded_imgs = self.autoencoder.decoder(encoded_imgs).numpy()

                # For improved accuracy use np.around()
                #reconstructed.append(np.around(np.squeeze(decoded_imgs)))
                reconstructed.append(np.squeeze(decoded_imgs))

            # Combining the different color channel images to one stacked image
            reconstructed = np.stack(reconstructed, axis=-1)

            # Using VerNet to get predictability and accuracy
            pred, acc = self.ver_net.check_predictability(reconstructed,
                                                          y_test,
                                                          tolerance=0.5)
            print("Predictability: " + str(pred) + ", accuracy:" + str(acc))

            self.show_figure(10, x_test, reconstructed, y_test, pred, acc)

        else:
            # Reshaping
            x_test = x_test[:, :, :, [0]]

            # Sending the images through the AE to get reconstructed images
            encoded_imgs = self.autoencoder.encoder(x_test).numpy()
            decoded_imgs = self.autoencoder.decoder(encoded_imgs).numpy()

            # Using VerNet to get predictability and accuracy
            pred, acc = self.ver_net.check_predictability(decoded_imgs, y_test)
            print("Predictability: " + str(pred) + ", accuracy:" + str(acc))

            self.show_figure(10, x_test, decoded_imgs, y_test, pred, acc)

    def show_figure(self, n, original, reconstructed, y_test, predictability,
                    accuracy):
        """
        Plotting original images and their reconstructions
        """

        # Showing the original images and reconstructed images
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(original[i].astype(np.float64))
            plt.title("Class " + str(y_test[i]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructed[i])
            plt.title("Reconstruct")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.suptitle("" + str(n) + " images reconstructed" +
                     " (Predictability: " + str(predictability) +
                     ", Accuracy: " + str(accuracy) + ")",
                     fontsize="x-large")

        # Choosing filepath
        if self.three_colors:
            path = "./results/ae-basic-color"
        else:
            path = "./results/ae-basic-mono"

        # Save figure
        if self.save_image:
            plt.savefig(path)

        # Show image
        plt.show()


if __name__ == "__main__":
    ae_basic = AEBasic(three_colors=True, save_image=False, save_weigths=False)

    ae_basic.run()