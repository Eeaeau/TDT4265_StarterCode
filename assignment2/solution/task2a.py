import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    X = X.astype(float)  #sol
    mean = 33.55  #sol
    std = 78.87  #sol
    X = (X - X.mean()) / X.std()  #sol
    ones = np.ones((X.shape[0], 1))  #sol
    X = np.concatenate((X, ones), axis=1)  #sol
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    ce = targets * np.log(outputs)  #sol
    return -ce.sum(axis=1).mean()  #sol
    raise NotImplementedError


def softmax(x: np.ndarray) -> np.ndarray:  #sol
    exp = np.exp(x)  #sol#sol
    a = exp / exp.sum(axis=1, keepdims=True)  #sol
    return a  #sol
#sol
#sol
# Improved sigmoid: a ta nh(bx) #sol
impr_sig_a = 1.7159  #sol
impr_sig_b = 2/3  #sol
def sigmoid(x, use_improved_sigmoid):  #sol
    if use_improved_sigmoid:  #sol
        return impr_sig_a * np.tanh(impr_sig_b * x)  #sol
    return 1 / (1 + np.exp(-x))  #sol
#sol
def sigmoid_prime(x, use_improved_sigmoid):  #sol
    if use_improved_sigmoid:  #sol
        return impr_sig_a * impr_sig_b * (1 - np.tanh(impr_sig_b*x)**2) #sol
        top = 2 * impr_sig_a * impr_sig_b  #sol
        denom = np.cosh(2 * impr_sig_b * x) + 1  #sol
        return top / denom  #sol
    sig = sigmoid(x, False)  #sol
    return sig * (1 - sig)  #sol
#sol
#sol
def weight_init(shape, use_improved_weight_init):  #sol
    if use_improved_weight_init:  #sol
        fan_in = shape[0]  #sol
        std = 1 / np.sqrt(fan_in) #sol
        return np.random.normal(scale=std, size=shape)  #sol
    return np.random.uniform(-1, 1, size=shape)  #sol
#sol
#sol
class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        np.random.seed(1) # Always reset random seed before weight init to get comparable results.        
        # Define number of input nodes
        self.I = None
        self.I = 785  #sol
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            w = weight_init(w_shape, use_improved_weight_init)  #sol
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
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        self.layer_inputs = []  #sol
        self.sigmoid_inputs = []  #sol
        for layer_idx in range(len(self.ws)):  #sol
            self.layer_inputs.append(X)  #sol
            w = self.ws[layer_idx]  #sol
            X = X.dot(w)  #sol
            if len(self.ws) - 1 == layer_idx:  # Last layer, softmax #sol
                X = softmax(X)  #sol
            else:  #sol
                self.sigmoid_inputs.append(X)  #sol
                X = sigmoid(X, self.use_improved_sigmoid)  #sol
#sol
        return X  #sol
        return None

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        self.grads = [None for i in range(len(self.ws))]  #sol
        delta = - (targets - outputs)  #sol
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []
        for layer_idx in range(len(self.ws) - 1, -1, -1):  #sol
            norm_factor = X.shape[0]  #sol
            dW = delta.T.dot(self.layer_inputs[layer_idx]) / norm_factor  #sol
            dW = dW.T #sol
            self.grads = [dW] + self.grads  #sol
            if layer_idx != 0:  #sol
                sigmoid_input = self.sigmoid_inputs[layer_idx-1]  #sol
                delta = sigmoid_prime( #sol
                    sigmoid_input, self.use_improved_sigmoid) * delta.dot(self.ws[layer_idx].T)  #sol

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

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
    # TODO: Implement this function (copy from last assignment)
    Y_oh = np.zeros((Y.shape[0], num_classes))  #sol
    Y_oh[range(len(Y)), Y.squeeze()] = 1  #sol
    return Y_oh  #sol
    raise NotImplementedError


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
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
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
