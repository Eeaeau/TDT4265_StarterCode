import numpy as np
import utils
import typing
import scipy.stats as stats

np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"
    X_train, _, X_val, _ = utils.load_full_mnist()
    # full_dataset = np.concatenate((X_train, X_val))

    X = stats.zmap(X, X_train, axis=None)
    bias = np.ones((X.shape[0], 1))
    X = np.hstack((X, bias))

    assert X.shape[1] == 785, f"X.shape[1]: {X.shape[1]}, should be 785"
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert (
        targets.shape == outputs.shape
    ), f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    batch_size = targets.size

    C = -targets.shape[1] / batch_size * np.tensordot(targets, np.log(outputs))

    return C


def sigmoid(X, use_improved_sigmoid=False):
    """
    Args:
        x: ndarray
    Returns:
        corresponding sigmoid value
    """
    if use_improved_sigmoid:
        return 1.7159 * np.tanh(2 / 3 * X)
    else:
        return 1 / (1 + np.exp(-X))


def sigmoid_prime(X, use_improved_sigmoid=False):
    if use_improved_sigmoid:
        # return 1.14393 / (np.cosh(2 * X / 3) ** 2)
        return 1.7159 * 2 / 3 * (1 - (np.tanh(2 * X / 3) ** 2))
    else:
        return sigmoid(X, False) * (1 - sigmoid(X, False))


def softmax(X: np.array):

    return np.exp(X) / np.sum(np.exp(X), axis=1)[:, None]


class SoftmaxModel:
    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3a hyperparameter
        use_improved_weight_init: bool,  # Task 3c hyperparameter
    ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer
        self.n_layers = len(self.neurons_per_layer)
        self.hidden_layer_output = []
        self.zs = []

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if use_improved_weight_init:
                w = np.random.normal(0, 1 / np.sqrt(prev), (w_shape))
            else:
                w = np.random.uniform(-1, 1, (w_shape))
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # Reset the lists
        Fs = X
        self.hidden_layer_output = [Fs]
        self.Zs = []
        # dont want to matmul the X with the weights for every layer, so need to store the last layer in a another variable and initiate that variable with X
        for i in range(
            len(self.neurons_per_layer) - 1
        ):  # -1 since we want the last layer to go through softmax
            Z = Fs @ self.ws[i]
            Fs = sigmoid(Z, self.use_improved_sigmoid)

            # saving the values
            self.Zs.append(Z)
            self.hidden_layer_output.append(Fs)

        # last layer need to be go through softmax
        Z = (
            Fs @ self.ws[-1]
        )  # then we have the last output nodes in Fs and can matmul them with the weights

        Y = softmax(Z)

        return Y

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []
        batch_size = X.shape[0]
        delta = -(targets - outputs)
        # defining the gradient for the first layer
        init_gradient = (self.hidden_layer_output[-1].T @ delta) / batch_size
        self.grads.append(init_gradient)
        for l in range(1, len(self.neurons_per_layer)):
            # print(l)
            z = self.Zs[-l]
            s = sigmoid_prime(z, self.use_improved_sigmoid)
            delta = (delta @ self.ws[-l].T) * s
            self.grads.insert(
                0, (self.hidden_layer_output[-l - 1].T @ delta) / batch_size
            )

        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    encoding = np.zeros((Y.size, num_classes))

    encoding[np.arange(Y.size), Y.ravel()] = 1  # inspired by keras

    return encoding


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited.
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon ** 2, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
