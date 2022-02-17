import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!
    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_momentum = False
    model_impw = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_impw = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_impw, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_impw, val_history_impw = trainer_impw.train(num_epochs)

    use_improved_sigmoid = True
    use_improved_weight_init = False
    use_momentum = False
    model_imps = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_imps = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_imps, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_imps, val_history_imps = trainer_imps.train(num_epochs)


    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False
    model_impsw = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_impsw = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_impsw, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_impsw, val_history_impsw = trainer_impsw.train(num_epochs)

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    model_impswm = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_impswm = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_impswm, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_impswm, val_history_impswm = trainer_impswm.train(num_epochs)

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    shuffle_data = False
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)
    shuffle_data = True

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_impw["loss"], "Task 3b Model - improved w", npoints_to_average=10)
    utils.plot_loss(
        train_history_imps["loss"], "Task 3a Model - improved sigmoid", npoints_to_average=10)
    utils.plot_loss(
        train_history_impsw["loss"], "Task 3b Model - improved sigmoid & w", npoints_to_average=10)

    utils.plot_loss(
        train_history_impswm["loss"], "Task 3c Model - improved sigmoid & w & momentum", npoints_to_average=10)
    utils.plot_loss(
        train_history_no_shuffle["loss"], "Task 2 Model - No dataset shuffling", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .95])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(
        val_history_impw["accuracy"], "Task 3b Model - improved w")
    utils.plot_loss(
        val_history_imps["accuracy"], "Task 3a Model - improved sigmoid")
    utils.plot_loss(
        val_history_impsw["accuracy"], "Task 3b Model - improved sigmoid & w")
    utils.plot_loss(
        val_history_impswm["accuracy"], "Task 3c Model - improved sigmoid & w & momentum")
    utils.plot_loss(
        val_history_no_shuffle["accuracy"], "Task 2 Model - No Dataset Shuffling")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task3c_all.png")
    plt.savefig("task3c_all_eps.eps")
    plt.show()
