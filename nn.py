from tensorflow import keras
import utils
import numpy as np
import matplotlib.pyplot as plt



def keras_model(num_hidden, learning_rate=0.01):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(num_hidden, input_shape=(56,), activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    opt = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer="SGD", loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
    return model


def train(xvals, yvals, range, num_hidden, num_epoch):
    loss_list = []
    for i in range:
        model = keras_model(num_hidden, learning_rate=i)
        model.fit(xvals, yvals, epochs=num_epoch, batch_size=1)
        loss, accuracy = model.evaluate(xvals, yvals)
        loss_list.append([i, loss, accuracy])
    return np.array(loss_list).transpose()


def ten_hidden(xvals, yvals):
    LR_range = np.logspace(-5, 1, num=30, base=2, dtype='float')
    loss = train(xvals, yvals, LR_range, 10, 10)
    min_index = np.argmin(loss[1])
    print("Minimum loss is {0} and accuracy is {1} with learning rate of {2}\n".format(
        loss[1][min_index],
        loss[2][min_index],
        loss[0][min_index]
    ))
    plt.plot(loss[0], loss[1], c="red", label="loss")


def main():
    xvals, yvals = utils.load_newts("", do_min_max=True)

    # 1. 10 hidden units
    ten_hidden(xvals, yvals)

    plt.show()

if __name__ == '__main__':
    main()
