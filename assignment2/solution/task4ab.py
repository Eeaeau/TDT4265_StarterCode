import utils
import matplotlib.pyplot as plt
from task2 import *
from task2a import SoftmaxModel

if __name__ == "__main__":
    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    # Hyperparameters

    # Hyperparameters
    num_epochs = 20
    learning_rate = .1 / 5
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9
#sol
    shuffle_data = True#sol
    use_improved_sigmoid = True#sol
    use_improved_weight_init = True#sol
    use_momentum = True#sol
    train_histories = []
    val_histories = []
    labels = []

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    TH, VH = trainer.train(num_epochs)
    train_histories.append(TH)
    val_histories.append(VH)
    labels.append("Baseline")
    neurons_per_layer = [32, 10]

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    TH, VH = trainer.train(num_epochs)
    train_histories.append(TH)
    val_histories.append(VH)
    labels.append("32 Hidden units")
    neurons_per_layer = [128, 10]
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    TH, VH = trainer.train(num_epochs)
    train_histories.append(TH)
    val_histories.append(VH)
    labels.append("128 hidden units")
    print(calculate_accuracy(X_val, Y_val, model))
#sol
     # Plot loss#sol
    plt.figure(figsize=(20, 12)) #sol
    plt.subplot(1, 2, 1)
    for H, label in zip(train_histories, labels):
        utils.plot_loss(H["loss"], label, npoints_to_average=10)
    plt.xlabel("Number of gradient steps")#sol
    plt.ylim([0, .6])
#    plt.xlim([1000, 7000])
    plt.ylabel("Training Cross Entropy Loss")#sol
    plt.legend()#sol
    plt.subplot(1, 2, 2)
    plt.ylim([0.9, 1.0])
    for H, label in zip(val_histories, labels):
        utils.plot_loss(H["accuracy"], label)
    plt.ylabel("Validation Accuracy")
    plt.legend()
    # plt.savefig("../latex/figures/task4ab_solution.png") #sol
    plt.savefig("task4ab_solution.png") #sol
    plt.show()