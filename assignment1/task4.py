from operator import mod
from statistics import mode
import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, one_hot_encode

from task3 import SoftmaxTrainer, SoftmaxModel, calculate_accuracy

np.random.seed(0)


def plot_figures(figures, nrows=1, ncols=1, size=(4, 4)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig = plt.figure(figsize=size)

    for i in range(1, nrows * ncols + 1):
        fig.add_subplot(nrows, ncols, i)
        plt.imshow(figures[i - 1])


if __name__ == "__main__":

    subtask = "b"  # adjust to run different subtasks

    # Train a model with L2 regularization (task 4b)

    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    if subtask == "b":
        model01 = SoftmaxModel(l2_reg_lambda=0.0)
        trainer01 = SoftmaxTrainer(
            model01,
            learning_rate,
            batch_size,
            shuffle_dataset,
            X_train,
            Y_train,
            X_val,
            Y_val,
        )
        train_history_reg01, val_history_reg01 = trainer01.train(num_epochs)

        model02 = SoftmaxModel(l2_reg_lambda=2.0)
        trainer02 = SoftmaxTrainer(
            model02,
            learning_rate,
            batch_size,
            shuffle_dataset,
            X_train,
            Y_train,
            X_val,
            Y_val,
        )
        train_history_reg02, val_history_reg02 = trainer02.train(num_epochs)

        # You can finish the rest of task 4 below this point.

        weights01 = model01.w[:-1, :]
        print(weights01.shape)
        weights01 = weights01.T.reshape((10, 28, 28))
        print(weights01.shape)
        # print(weights)
        # fig = plt.figure(figsize=(1, 10))

        # for weigth in weights:
        #     img = plt.imshow(weigth)

        # plot_figures(weights01, 1, 11, (15, 4))
        # plt.show()

        weights02 = model02.w[:-1, :]
        print(weights02.shape)
        weights02 = weights02.T.reshape((10, 28, 28))
        print(weights02.shape)

        # plot_figures(weights02, 1, 11, (15, 4))
        plot_figures(np.concatenate((weights01, weights02), axis=0), 2, 10, (15, 4))
        # # Plotting of softmax weights (Task 4b)
        # plt.imsave("task4b_softmax_weight.eps", weights01, cmap="gray")
        plt.show()

    elif (subtask == "c") | (subtask == "d"):
        # ----- Plotting of accuracy for difference values of lambdas (task 4c) ----- #
        l2_lambdas = [2, 0.2, 0.02, 0.002]
        L2_norms = []

        for l2_lambda in l2_lambdas:

            model2 = SoftmaxModel(l2_reg_lambda=l2_lambda)

            trainer = SoftmaxTrainer(
                model2,
                learning_rate,
                batch_size,
                shuffle_dataset,
                X_train,
                Y_train,
                X_val,
                Y_val,
            )
            train_history_reg02, val_history_reg02 = trainer.train(num_epochs)
            L2_norm = np.linalg.norm(model2.w)
            L2_norms.append(L2_norm)

            # # Plot accuracy
            # # plt.ylim([0.89, 0.93])
            utils.plot_loss(
                val_history_reg02["accuracy"],
                "Validation Accuracy with $\lambda$=" + str(l2_lambda),
            )

        plt.xlabel("Number of Training Steps")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig("task4c_l2_reg_accuracy.eps")
        plt.show()

        if subtask == "d":
            # Task 4d - Plotting of the l2 norm for each weight

            plt.plot(l2_lambdas, L2_norms)

            plt.xlabel("$\lambda$")
            plt.ylabel("$L_2$ norm")

            plt.savefig("task4d_l2_reg_norms.eps")
            plt.show()
