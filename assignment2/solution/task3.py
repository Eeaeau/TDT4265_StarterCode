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
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(num_epochs)
    shuffle_data = True

    plt.figure(figsize=(20, 12)) #sol
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"], "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(train_history_no_shuffle["loss"], "Task 2 Model - No dataset shuffling", npoints_to_average=10)
    plt.xlabel("Number of gradient steps")#sol
    plt.ylim([0, .4])
    plt.ylabel("Training Cross Entropy Loss")#sol
    plt.legend()#sol
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .95])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(val_history_no_shuffle["accuracy"], "Task 2 Model - No Dataset Shuffling")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("../latex/figures/task3_shuffle_example.png") #sol
    plt.show()
    # Task 3 #sol
    train_histories = [train_history]  # sol
    val_histories = [val_history]  # sol
    labels = ["Train Loss"]  # sol
    use_improved_weight_init = True  # sol
  # sol
    model = SoftmaxModel(  # sol
        neurons_per_layer,  # sol
        use_improved_sigmoid,  # sol
        use_improved_weight_init)  # sol
    trainer = SoftmaxTrainer(  # sol
        momentum_gamma, use_momentum,  # sol
        model, learning_rate, batch_size, shuffle_data,  # sol
        X_train, Y_train, X_val, Y_val,  # sol
    )  # sol
    TH, VH = trainer.train(num_epochs)  # sol
    train_histories.append(TH)  # sol
    val_histories.append(VH)  # sol
    labels.append(f"init={use_improved_weight_init}, sigmoid={use_improved_sigmoid}, momentum={use_momentum}")  # sol
  # sol
    use_improved_sigmoid = True  # sol
    model = SoftmaxModel(  # sol
        neurons_per_layer,  # sol
        use_improved_sigmoid,  # sol
        use_improved_weight_init)  # sol
    trainer = SoftmaxTrainer(  # sol
        momentum_gamma, use_momentum,  # sol
        model, learning_rate, batch_size, shuffle_data,  # sol
        X_train, Y_train, X_val, Y_val,  # sol
    )  # sol
    TH, VH = trainer.train(num_epochs)  # sol
    train_histories.append(TH)  # sol
    val_histories.append(VH)  # sol
    labels.append(f"init={use_improved_weight_init}, sigmoid={use_improved_sigmoid}, momentum={use_momentum}")  # sol
  # sol
    use_momentum = True  # sol
    learning_rate = 0.02  # sol
    model = SoftmaxModel(  # sol
        neurons_per_layer,  # sol
        use_improved_sigmoid,  # sol
        use_improved_weight_init)  # sol
    trainer = SoftmaxTrainer(  # sol
        momentum_gamma, use_momentum,  # sol
        model, learning_rate, batch_size, shuffle_data,  # sol
        X_train, Y_train, X_val, Y_val,  # sol
    )  # sol
    TH, VH = trainer.train(num_epochs)  # sol
    train_histories.append(TH)  # sol
    val_histories.append(VH)  # sol
    labels.append(f"init={use_improved_weight_init}, sigmoid={use_improved_sigmoid}, momentum={use_momentum}")  # sol
  # sol
    plt.figure(figsize=(20, 12)) #sol  # sol
    plt.subplot(1, 2, 1)  # sol
    for H, label in zip(train_histories, labels):  # sol
        utils.plot_loss(H["loss"], label, npoints_to_average=10)  # sol
    plt.xlabel("Number of gradient steps")#sol  # sol
    plt.ylim([0, .6])  # sol
    plt.ylabel("Training Cross Entropy Loss")#sol  # sol
    plt.legend()#sol  # sol
    plt.subplot(1, 2, 2)  # sol
    plt.ylim([0.9, 1.0])  # sol
    for H, label in zip(val_histories, labels):  # sol
        utils.plot_loss(H["accuracy"], label)  # sol
    plt.ylabel("Validation Accuracy")  # sol
    plt.legend()  # sol
    # plt.savefig("../latex/figures/task3_solution.png") #sol  # sol
    plt.savefig("task3_solution.png") #sol  # sol
    plt.show()  # sol
