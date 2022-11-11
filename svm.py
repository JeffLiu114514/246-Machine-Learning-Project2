from sklearn import svm
from sklearn.svm import LinearSVC
import utils
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# SVM 1. LinearSVC + slack variable parameter
def LinearSVC_train(x_train, y_train, slack_param=1.0):
    linear_classifier = LinearSVC(C=slack_param, max_iter=10 ** 8)
    linear_classifier.fit(x_train, y_train)
    return linear_classifier


def linear_svm_eval_training_set(xvals, yvals, slack_param):
    linear_classifier = LinearSVC_train(xvals, yvals, slack_param)
    predictions = linear_classifier.predict(xvals)
    return f1_score(yvals, predictions), accuracy_score(yvals, predictions)


def linear_svm_eval_kfold_stub(xvals, yvals, slack_param, fold):
    kf = KFold(n_splits=fold, shuffle=True)
    f1scores, accuracyscores = [], []
    for train, test in kf.split(xvals):
        xtrain_this_fold, xtest_this_fold = xvals[train], xvals[test]
        ytrain_this_fold, ytest_this_fold = yvals[train], yvals[test]
        # train a model on this fold
        linear_classifier = LinearSVC_train(xtrain_this_fold, ytrain_this_fold, slack_param)
        # test the model on this fold
        predictions = linear_classifier.predict(xtest_this_fold)
        f1scores.append(f1_score(ytest_this_fold, predictions))
        accuracyscores.append(accuracy_score(ytest_this_fold, predictions))
    return sum(f1scores) / fold, sum(accuracyscores) / fold


def linear_svm_loop_slack(xvals, yvals, fold, c_range):
    scores = []
    for i in c_range:
        training_set_f1_score, training_set_accuracy_score = linear_svm_eval_training_set(xvals, yvals, i)
        cross_validation_f1_score, cross_validation_accuracy_score = linear_svm_eval_kfold_stub(xvals, yvals, i, fold)
        scores.append([i, training_set_f1_score, training_set_accuracy_score, cross_validation_f1_score,
                       cross_validation_accuracy_score])

    return np.array(scores).transpose()


def linear_svm_final(xvals, yvals):
    fold = 5
    c_range = np.logspace(-5, 3, num=101, base=2, dtype='float')
    scores = linear_svm_loop_slack(xvals, yvals, fold, c_range)
    print("F1 score on training set: {f1train} with parameter {p1} \n"
          "accuracy score on training set: {acctrain} with parameter {p2} \n"
          "F1 score on cross validation: {f1cross} with parameter {p3} \n"
          "accuracy score on cross validation: {acccross} with parameter {p4} \n".format(
        f1train=max(scores[1]), p1=scores[0][np.argmax(scores[1])],
        acctrain=max(scores[2]), p2=scores[0][np.argmax(scores[2])],
        f1cross=max(scores[3]), p3=scores[0][np.argmax(scores[3])],
        acccross=max(scores[4]), p4=scores[0][np.argmax(scores[4])]))
    # plt.xscale('log')
    plt.plot(scores[0], scores[1], c="red", label="F1 training dataset")
    plt.plot(scores[0], scores[3], c="green", label="F1 K-fold")
    # plt.plot(scores[0], scores[2], c="blue", label="accuracy training dataset")
    # plt.plot(scores[0], scores[4], c="yellow", label="accuracy K-fold")
    plt.title("Linear SVM")
    plt.xlabel("Slack values")
    plt.ylabel("F1 score")
    plt.legend()
    plt.show()


# SVM 2. kbf kernel + slack variable parameter
def SVC_train(x_train, y_train, slack_param=1.0, scale_param='scale'):
    classifier = svm.SVC(kernel='rbf', C=slack_param, gamma=scale_param)
    classifier.fit(x_train, y_train)
    return classifier


def SVC_eval_training_set(xvals, yvals, slack_param=1.0, scale_param='scale'):
    classifier = SVC_train(xvals, yvals, slack_param=slack_param, scale_param=scale_param)
    predictions = classifier.predict(xvals)
    return f1_score(yvals, predictions), accuracy_score(yvals, predictions)


def SVC_eval_kfold_stub(xvals, yvals, fold, slack_param=1.0, scale_param='scale'):
    kf = KFold(n_splits=fold, shuffle=True)
    f1scores, accuracyscores = [], []
    for train, test in kf.split(xvals):
        xtrain_this_fold, xtest_this_fold = xvals[train], xvals[test]
        ytrain_this_fold, ytest_this_fold = yvals[train], yvals[test]
        # train a model on this fold
        classifier = SVC_train(xtrain_this_fold, ytrain_this_fold, slack_param=slack_param, scale_param=scale_param)
        # test the model on this fold
        predictions = classifier.predict(xtest_this_fold)
        f1scores.append(f1_score(ytest_this_fold, predictions))
        accuracyscores.append(accuracy_score(ytest_this_fold, predictions))
    return sum(f1scores) / fold, sum(accuracyscores) / fold


def SVC_loop_slack(xvals, yvals, fold, c_range):
    scores = []
    for i in c_range:
        training_set_f1_score, training_set_accuracy_score = SVC_eval_training_set(xvals, yvals, slack_param=i)
        cross_validation_f1_score, cross_validation_accuracy_score = SVC_eval_kfold_stub(xvals, yvals, fold, slack_param=i)
        scores.append([i, training_set_f1_score, training_set_accuracy_score, cross_validation_f1_score,
                       cross_validation_accuracy_score])

    return np.array(scores).transpose()


def SVC_final_slack(xvals, yvals):
    fold = 5
    # c_range = np.logspace(-5, 3, num=101, base=2, dtype='float')
    c_range = np.linspace(0.1, 50, num=101, dtype='float')
    scores = SVC_loop_slack(xvals, yvals, fold, c_range)
    print("F1 score on training set: {f1train} with parameter {p1} \n"
          "accuracy score on training set: {acctrain} with parameter {p2} \n"
          "F1 score on cross validation: {f1cross} with parameter {p3} \n"
          "accuracy score on cross validation: {acccross} with parameter {p4} \n".format(
        f1train=max(scores[1]), p1=scores[0][np.argmax(scores[1])],
        acctrain=max(scores[2]), p2=scores[0][np.argmax(scores[2])],
        f1cross=max(scores[3]), p3=scores[0][np.argmax(scores[3])],
        acccross=max(scores[4]), p4=scores[0][np.argmax(scores[4])]))
    # plt.xscale('log')
    plt.plot(scores[0], scores[1], c="red", label="F1 training dataset")
    plt.plot(scores[0], scores[3], c="green", label="F1 K-fold")
    # plt.plot(scores[0], scores[2], c = "blue", label = "accuracy training dataset")
    # plt.plot(scores[0], scores[4], c = "yellow", label = "accuracy K-fold")
    plt.title("SVC rbf")
    plt.xlabel("Slack values")
    plt.ylabel("F1 score")
    plt.legend()
    plt.show()


# SVM 3. kbf kernel + scale parameter
def SVC_loop_scale(xvals, yvals, fold, scale_range):
    scores = []
    for i in scale_range:
        training_set_f1_score, training_set_accuracy_score = SVC_eval_training_set(xvals, yvals, scale_param=i)
        cross_validation_f1_score, cross_validation_accuracy_score = SVC_eval_kfold_stub(xvals, yvals, fold, scale_param=i)
        scores.append([i, training_set_f1_score, training_set_accuracy_score, cross_validation_f1_score,
                       cross_validation_accuracy_score])
    return np.array(scores).transpose()


def SVC_final_scale(xvals, yvals):
    fold = 5
    # c_range = np.logspace(-5, 3, num=101, base=2, dtype='float')
    scale_range = np.linspace(0.1, 4, num=101, dtype='float')
    scores = SVC_loop_scale(xvals, yvals, fold, scale_range)
    print("F1 score on training set: {f1train} with parameter {p1} \n"
          "accuracy score on training set: {acctrain} with parameter {p2} \n"
          "F1 score on cross validation: {f1cross} with parameter {p3} \n"
          "accuracy score on cross validation: {acccross} with parameter {p4} \n".format(
        f1train=max(scores[1]), p1=scores[0][np.argmax(scores[1])],
        acctrain=max(scores[2]), p2=scores[0][np.argmax(scores[2])],
        f1cross=max(scores[3]), p3=scores[0][np.argmax(scores[3])],
        acccross=max(scores[4]), p4=scores[0][np.argmax(scores[4])]))
    # plt.xscale('log')
    plt.plot(scores[0], scores[1], c="red", label="F1 training dataset")
    plt.plot(scores[0], scores[3], c="green", label="F1 K-fold")
    # plt.plot(scores[0], scores[2], c = "blue", label = "accuracy training dataset")
    # plt.plot(scores[0], scores[4], c = "yellow", label = "accuracy K-fold")
    plt.title("SVC rbf")
    plt.xlabel("Scale values")
    plt.ylabel("F1 score")
    plt.legend()
    plt.show()


def main():
    xvals, yvals = utils.load_newts("", do_min_max=True)
    linear_svm_final(xvals, yvals)  # question1
    SVC_final_slack(xvals, yvals)  # question2
    SVC_final_scale(xvals, yvals)  # question3


if __name__ == '__main__':
    main()
