import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float: 
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    accuracy = 0
    logits = model.forward(X) #sol
    preds = logits.argmax(axis=1)#sol
    y = targets.argmax(axis=1)#sol
    assert y.shape == preds.shape#sol
    correct_preds = preds == y#sol
    accuracy = correct_preds.mean()#sol
    return accuracy


class SoftmaxTrainer(BaseTrainer):

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
        # TODO: Implement this function (task 3b)
        loss = 0
        logits = self.model.forward(X_batch) #sol
        self.model.backward(X_batch, logits, Y_batch) #sol
        self.model.w = self.model.w - self.model.grad * self.learning_rate #sol
        self.model.zero_grad() #sol
        loss = cross_entropy_loss(Y_batch, logits) #sol
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
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    num_epochs = 500 #sol
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

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
        cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
        cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))


    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"], "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.6, .99])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()


    # Train a model with L2 regularization (task 4b)

    weight1 = model.w #sol
    model1 = SoftmaxModel(l2_reg_lambda=2.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    weight2 = model1.w #sol
    print("Final Test accuracy:", calculate_accuracy(X_val, Y_val, model1))#sol
    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)
    norm = lambda x: (x - x.min()) / (x.max() - x.min())#sol
    reshape = lambda x: np.concatenate(x[:-1].T.reshape(10, 28, 28), axis=1)#sol
    #sol
    weight1 = reshape(weight1)#sol
    weight1 = norm(weight1)#sol
    weight2 = reshape(weight2)#sol
    weight2 = norm(weight2)#sol
    weight = np.concatenate((weight1, weight2))#sol
    plt.imsave("task4b_softmax_weight.png", weight, cmap="gray")
    plt.imshow(weight)#sol
    plt.show()#sol

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [2, .2, .02, .002]
    l2_norms = {}#sol
    for l2_reg_lambda in l2_lambdas:#sol
        model = SoftmaxModel(l2_reg_lambda)#sol
        trainer = SoftmaxTrainer(#sol
            model, learning_rate, batch_size, shuffle_dataset,#sol
            X_train, Y_train, X_val, Y_val,#sol
        )#sol
        _, val_history = trainer.train(50)#sol
        utils.plot_loss(val_history["accuracy"], f"Validation accuracy. Lambda={l2_reg_lambda}")#sol
        l2_norms[l2_reg_lambda] = (model.w ** 2).sum()#sol
    plt.legend()#sol
    plt.ylim([0.7, .99]) #sol
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show() #sol

    # Task 4d - Plotting of the l2 norm for each weight

    plt.plot(range(len(l2_norms)), list(l2_norms.values()))#sol
    plt.xticks(range(len(l2_norms)), list(l2_norms.keys()))#sol
    plt.xlabel("lamda")#sol
    plt.ylabel("L2 norm")#sol
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()#sol
    #sol
