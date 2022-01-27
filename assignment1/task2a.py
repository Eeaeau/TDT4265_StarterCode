from matplotlib import axes
import numpy as np
import scipy.stats as stats
import utils
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] in the range (-1, 1)
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)

    # centering around zero and normalizing by std
    # X = stats.zscore(X)

    # could use numpy.linalg.norm

    # X = np.concatenate(([1], X), axis=0) # appears to be the fastest method https://stackoverflow.com/questions/36998260/prepend-element-to-numpy-array
    print('shape:', X.shape)
    # X = np.concatenate((X, 1), axis=0) # appears to be the fastest method https://stackoverflow.com/questions/36998260/prepend-element-to-numpy-array
    # X = np.append(X, [1])
    # X_norm = np.ones((X.shape[0], X.shape[1]+1))

    bias = np.ones(X.shape[0])

    X_norm = np.concatenate((X, bias.T), axis=1)

    # centering around zero and normalizing
    # X_norm[:-1] = X_norm[:-1] - float(np.mean(X))
    # X_norm[:-1] = X_norm[:-1] / np.max(np.abs(X))

    # X_norm[:,-1] = 1.0

    return X_norm

    # return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, 1]
        outputs: outputs of model of shape: [batch size, 1]
    Returns:
        Cross entropy error (float)
    """
    # TODO implement this function (Task 2a)

    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    N = targets.shape(0)
    # Cn = np.empty(N)
    # for n in range(N):
    #     y = targets[n]
    #     y_hat = outputs[n]

    #     Cn[n] = -(y*np.log(y_hat)+(1+y)*np.log(1-y_hat))


    C = -(targets*np.log(outputs)+(1+targets)*np.log(1-outputs))
    return 1/N*C

def sigmoid(x):
    """
    Args:
        x: float
    Returns:
        corresponding sigmoid value
    """
    return 1/(1 + np.exp(-x))

class BinaryModel:

    def __init__(self):
        # Define number of input nodes
        self.I = 785
        self.w = np.zeros((self.I, 1))
        self.grad = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, 1]
        """
        # TODO implement this function (Task 2a)
        batch_size = X.shape(0)
        y = np.empty(batch_size)

        for i in range (batch_size):
            y[i] = sigmoid(self.w.dot(X[i]))

        return y

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, 1]
            targets: labels/targets of each image of shape: [batch size, 1]
        """
        # TODO implement this function (Task 2a)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        self.grad = np.zeros_like(self.w)
        assert self.grad.shape == self.w.shape,\
            f"Grad shape: {self.grad.shape}, w: {self.w.shape}"

        batch_size = X.shape(0)
        outputs = np.empty(batch_size)

        print('grad ', self.grad.shape)
        for n in range(batch_size):
            self.grad += -1/self.I*(targets-outputs)*X

        self.grad /= batch_size

        nabla = 1
        print('X ', X.shape)
        print('outputs ', outputs.shape)
        print('targets ', targets.shape)

        self.w = self.w-nabla*self.grad
        print('w ', self.w.shape)

    def zero_grad(self) -> None:
        self.grad = None


def gradient_approximation_test(model: BinaryModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited.
        Details about this test is given in the appendix in the assignment.
    """
    w_orig = np.random.normal(loc=0, scale=1/model.w.shape[0]**2, size=model.w.shape)
    epsilon = 1e-3
    for i in range(w_orig.shape[0]):
        model.w = w_orig.copy()
        orig = w_orig[i].copy()
        model.w[i] = orig + epsilon
        logits = model.forward(X)
        cost1 = cross_entropy_loss(Y, logits)
        model.w[i] = orig - epsilon
        logits = model.forward(X)
        cost2 = cross_entropy_loss(Y, logits)
        gradient_approximation = (cost1 - cost2) / (2 * epsilon)
        model.w[i] = orig
        # Actual gradient
        logits = model.forward(X)
        model.backward(X, logits, Y)
        difference = gradient_approximation - model.grad[i, 0]
        assert abs(difference) <= epsilon**2,\
            f"Calculated gradient is incorrect. " \
            f"Approximation: {gradient_approximation}, actual gradient: {model.grad[i,0]}\n" \
            f"If this test fails there could be errors in your cross entropy loss function, " \
            f"forward function or backward function"


if __name__ == "__main__":
    category1, category2 = 2, 3
    X_train, Y_train, *_ = utils.load_binary_dataset(category1, category2)
    X_train = pre_process_images(X_train)
    assert X_train.max() <= 1.0, f"The images (X_train) should be normalized to the range [-1, 1]"
    assert X_train.min() < 0 and X_train.min() >= -1, f"The images (X_train) should be normalized to the range [-1, 1]"
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Simple test for forward pass. Note that this does not cover all errors!
    model = BinaryModel()
    logits = model.forward(X_train)
    np.testing.assert_almost_equal(
        logits.mean(), .5,
        err_msg="Since the weights are all 0's, the sigmoid activation should be 0.5")

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for i in range(2):
        gradient_approximation_test(model, X_train, Y_train)
        model.w = np.random.randn(*model.w.shape)
