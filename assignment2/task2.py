import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
from trainer import BaseTrainer

np.random.seed(0)


def calculate_accuracy(
    X: np.ndarray, targets: np.ndarray, model: SoftmaxModel
) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (copy from last assignment)
    # pred_index = np.argmax(
    #     model.forward(X), axis=1
    # )  # since we are using softmax, the argmax will be the index with the highest value.
    # since targets are one the form [0,0,0,1,0,0,0,0,0,0] (here ex 3) (from one hot encoding) using argmax on it will give us the correct index. Then we can compare the index that are
    # alike and count them.
    accuracy = 0
    logits = model.forward(X)
    preds = logits.argmax(axis=1)
    y = targets.argmax(axis=1)
    assert y.shape == preds.shape
    correct_preds = preds == y
    accuracy = correct_preds.mean()
    return accuracy


class SoftmaxTrainer(BaseTrainer):
    def __init__(
        self, momentum_gamma: float, use_momentum: bool, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.momentum_gamma = momentum_gamma
        self.use_momentum = use_momentum
        # Init a history of previous gradients to use for implementing momentum
        self.previous_grads = [np.zeros_like(w) for w in self.model.ws]

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 2c)

        logits = self.model.forward(X_batch)
        self.model.backward(X_batch, logits, Y_batch)

        if self.use_momentum:
            for idx, w in enumerate(self.model.ws):
                w -= self.learning_rate * self.previous_grads[idx]

                self.previous_grads[idx] = (
                    self.model.grads[idx]
                    + self.momentum_gamma * self.previous_grads[idx]
                )
        else:
            for idx, w in enumerate(self.model.ws):
                w -= self.learning_rate * self.model.grads[idx]

        self.model.zero_grad()
        loss = cross_entropy_loss(Y_batch, logits)  # sol? nope its snowing now

        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = 0.9  # Task 3 hyperparameter
    shuffle_data = True

    # Settings for task 3. Keep all to false for task 2.
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    # Hyperparameters

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

    print(
        "Final Train Cross Entropy Loss:",
        cross_entropy_loss(Y_train, model.forward(X_train)),
    )
    print(
        "Final Validation Cross Entropy Loss:",
        cross_entropy_loss(Y_val, model.forward(X_val)),
    )
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    # Plot loss for first model (task 2c)
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0.0, 0.5])
    utils.plot_loss(train_history["loss"], "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.90, 0.99])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    # plt.savefig("task2c_train_loss.png")
    plt.savefig("task2c_train_loss.eps")
    plt.show()
