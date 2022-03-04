from stacked_mnist import DataMode, StackedMNISTData

gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE,
                       default_batch_size=2048)
# Getting training and test data
x_train, y_train = gen.get_full_data_set(training=True)
x_test, y_test = gen.get_full_data_set(training=False)

# Reshaping
x_train = x_train[:, :, :, [0]]
x_test = x_test[:, :, :, [0]]
