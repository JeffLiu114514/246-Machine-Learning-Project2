from tensorflow import keras
import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


def keras_model(num_hidden, learning_rate=0.01):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(num_hidden, input_shape=(56,), activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    opt = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer="SGD", loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
    return model


def train_model(xvals, yvals, learning_rate, num_hidden, num_epoch):
    # The annotated iteration below tested that learning rate of 0.20 could reach 100% accuracy in 200 epochs
    # loss_list = []
    # for i in learning_rate:
    #     model = keras_model(num_hidden, learning_rate=i)
    #     model.fit(xvals, yvals, epochs=num_epoch, batch_size=1)
    #     loss, accuracy = model.evaluate(xvals, yvals)
    #     loss_list.append([i, loss, accuracy])
    # return np.array(loss_list).transpose()
    model = keras_model(num_hidden, learning_rate=learning_rate)
    model.fit(xvals, yvals, epochs=num_epoch, batch_size=1)
    return model


def ten_hidden(xvals, yvals):
    learning_rate = 0.20  # np.linspace(0.01, 1, num=101, dtype='float')
    num_hidden = 10
    num_epoch = 200
    model = train_model(xvals, yvals, learning_rate, num_hidden, num_epoch)
    loss, accuracy = model.evaluate(xvals, yvals)
    print("Minimum loss is {0} and accuracy is {1} with learning rate of 0.25\n".format(
        loss,
        accuracy
    ))
    # min_index = np.argmin(loss[1])
    # print("Minimum loss is {0} and accuracy is {1} with learning rate of {2}\n".format(
    #     loss[1][min_index],
    #     loss[2][min_index],
    #     loss[0][min_index]
    # ))
    # plt.plot(loss[0], loss[1], c="red", label="loss")


def to_binary(list):
    binary_list = []
    for output in list.flatten():
        if output >= 0.5:
            binary_list.append(1)
        else:
            binary_list.append(0)
    return binary_list


def training_dataset(xvals, yvals, num_hidden):
    num_epoch = 200
    learning_rate = 0.20
    model = train_model(xvals, yvals, learning_rate, num_hidden, num_epoch)
    predictions = model.predict(xvals)
    binary_predictions = to_binary(predictions)
    return f1_score(yvals, binary_predictions)


def cross_validation(xvals, yvals, num_hidden):
    fold = 5
    num_epoch = 200
    learning_rate = 0.20
    kf = KFold(n_splits=fold, shuffle=True)
    f1 = []
    for train, test in kf.split(xvals):
        xtrain_this_fold, xtest_this_fold = xvals[train], xvals[test]
        ytrain_this_fold, ytest_this_fold = yvals[train], yvals[test]
        model = train_model(xtrain_this_fold, ytrain_this_fold, learning_rate, num_hidden, num_epoch)
        predictions = model.predict(xtest_this_fold)
        binary_predictions = to_binary(predictions)
        f1.append(f1_score(ytest_this_fold, binary_predictions))
    return sum(f1) / fold


def sweep_hidden(xvals, yvals):
    hidden_units_range = np.linspace(1, 10, num=10, dtype='int')
    kfold, training = [], []
    for i in hidden_units_range:
        f1_training = training_dataset(xvals, yvals, i)
        training.append([i, f1_training])
        f1_cross = cross_validation(xvals, yvals, i)
        kfold.append([i, f1_cross])
        print("\nHidden unit {0} finished.\n".format(i))
    return np.array(training).transpose(), np.array(kfold).transpose()


def main():
    xvals, yvals = utils.load_newts("", do_min_max=True)

    # 1. 10 hidden units
    # ten_hidden(xvals, yvals)

    # 3.sweeping the number of hidden units from 1 to 10
    training, kfold = sweep_hidden(xvals, yvals)
    plt.plot(training[0], training[1], c="red", label="F1 training")
    plt.plot(kfold[0], kfold[1], c="green", label="F1 K-fold")
    plt.title("MLP")
    plt.xlabel("# of hidden units")
    plt.ylabel("F1 score")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
