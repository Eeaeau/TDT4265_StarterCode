import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = 0.9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    learning_rate = 0.02

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init
    )
    trainer = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model,
        learning_rate,
        batch_size,
        shuffle_data,
        X_train,
        Y_train,
        X_val,
        Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    # ---------------------- Improved weights --------------------- #
    use_improved_weight_init = True

    model_improved_weight = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init
    )
    trainer_improved_weight = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model_improved_weight,
        learning_rate,
        batch_size,
        shuffle_data,
        X_train,
        Y_train,
        X_val,
        Y_val,
    )

    (
        train_history_improved_weight,
        val_history_improved_weight,
    ) = trainer_improved_weight.train(num_epochs)

    # ---------------------- improved_sigmoid --------------------- #

    use_improved_sigmoid = True

    model_improved_sigmoid = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init
    )
    trainer_improved_sigmoid = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model_improved_sigmoid,
        learning_rate,
        batch_size,
        shuffle_data,
        X_train,
        Y_train,
        X_val,
        Y_val,
    )

    (
        train_history_improved_sigmoid,
        val_history_improved_sigmoid,
    ) = trainer_improved_sigmoid.train(num_epochs)

    # ---------------------- momentum_gamma --------------------- #

    use_momentum = True

    model_momentum_gamma = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init
    )
    trainer_momentum_gamma = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model_momentum_gamma,
        learning_rate,
        batch_size,
        shuffle_data,
        X_train,
        Y_train,
        X_val,
        Y_val,
    )

    (
        train_history_momentum_gamma,
        val_history_momentum_gamma,
    ) = trainer_momentum_gamma.train(num_epochs)

    plt.subplot(1, 2, 1)
    plt.ylabel("Train loss")
    utils.plot_loss(val_history["loss"], "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(
        val_history_improved_weight["loss"],
        "Task 2 Model - Added improved weights",
        npoints_to_average=10,
    )
    utils.plot_loss(
        val_history_improved_sigmoid["loss"],
        "Task 2 Model - Added improved sigmoid",
        npoints_to_average=10,
    )
    utils.plot_loss(
        val_history_momentum_gamma["loss"],
        "Task 2 Model - Added momentum gamma",
        npoints_to_average=10,
    )

    plt.ylim([0, 0.4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1.0])

    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(
        train_history_improved_weight["accuracy"],
        "Task 2 Model - Added improved weights",
    )
    utils.plot_loss(
        train_history_improved_sigmoid["accuracy"],
        "Task 2 Model - Added improved sigmoid",
    )
    utils.plot_loss(
        train_history_momentum_gamma["accuracy"], "Task 2 Model - Added momentum gamma"
    )

    plt.ylabel("Validation Accuracy")
    plt.legend()

    # if use_momentum:
    #     plt.savefig("task3c_train_loss.eps")
    # elif use_improved_sigmoid:
    #     plt.savefig("task3b_train_loss.eps")
    # elif use_improved_weight_init:

    plt.rcParams["figure.figsize"] = (50, 3)
    plt.savefig("task3_train_loss.eps")
    plt.show()
