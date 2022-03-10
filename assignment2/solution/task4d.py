import utils
from task2 import *
from task2a import SoftmaxModel


if __name__ == "__main__":
    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    num_epochs = 10
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True
#sol
    # Settings for task 3. Keep all to false for task 2.
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

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

    neurons_per_layer = [128, 128, 10]
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_2, val_history_2 = trainer.train(num_epochs)
    
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_5, val_history_5 = trainer.train(num_epochs)

#sol
     # Plot loss#sol

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.ylim([0.0, 1])#sol
    utils.plot_loss(train_history["loss"], "Train Loss", npoints_to_average=10)#sol
    utils.plot_loss(train_history_2["loss"], "Train 3 layers 59 units", npoints_to_average=10)#sol
    utils.plot_loss(train_history_5["loss"], "Train 10 layers 64 units", npoints_to_average=10)#sol
#sol
    plt.xlabel("Number of gradient steps")#sol
    plt.ylabel("Cross Entropy Loss")#sol
    plt.legend()#sol
    plt.subplot(1, 2, 2)
    plt.ylim([0.9, 1.0])
    utils.plot_loss(val_history["accuracy"], "Validation ")#sol
    utils.plot_loss(val_history_2["accuracy"], "Validation 3 layers 59 units")#sol
    utils.plot_loss(val_history_5["accuracy"], "Validation 10 layers 64 units")#sol
    plt.xlabel("Number of gradient steps")#sol
    plt.ylabel("Accuracy")#sol
    plt.legend()#sol
    # plt.savefig("../latex/figures/task4d_solution.png") #sol
    plt.savefig("task4d_solution.png") #sol
    plt.show() #sol