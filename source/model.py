import tqdm
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict
from operator import itemgetter

from text_preprocesssing import preprocess_corpus
from globals import CORPUS_PATH, VECTOR_SPACE_SIZE, OPTIMIZER, LEARN_RATE, EPOCHS


def relu(tensor: np.ndarray) -> np.ndarray:
    """
    Does the relu function.
    f(x) = max(0, x)
    """
    return tensor * (tensor > 0.)


def relu_derivative(tensor: np.ndarray) -> np.ndarray:
    return 1. * (tensor > 0.)


def softmax(tensor: np.ndarray) -> np.ndarray:
    """
    Does the safe softmax function with numerical stability
    """
    mxs = np.max(tensor, axis=1).reshape(-1, 1)
    exps = np.exp(tensor - mxs)

    sums = np.sum(exps, axis=1).reshape(-1, 1)

    return exps / sums


def make_network_hyperparameters(input_size: int, vector_space_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes random weights and biases for the network
    Returns a pair of weights, input-hidden and hidden-output
    """
    input_2_hidden = np.random.uniform(-1, 1, size=(input_size, vector_space_size))
    hidden_2_output = np.random.uniform(-1, 1, size=(vector_space_size, input_size))
    return input_2_hidden, hidden_2_output


def forward_one_word(word_vector: np.ndarray, network_hyperparams: Tuple[np.ndarray, np.ndarray])\
        -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Does the forward pass for one word of the network.
    Network hyperparameters are in a tuple of input to hidden and hidden to output weights

    Returns a tuple with all the transfer and activation pairs (in order of layers),
    with the last layers' activation being the output
    """
    input_2_hidden_weights, hidden_2_output_weights = network_hyperparams

    input_2_hidden_transfer = word_vector @ input_2_hidden_weights
    input_2_hidden_activation = relu(input_2_hidden_transfer)

    hidden_2_output_transfer = input_2_hidden_activation @ hidden_2_output_weights
    hidden_2_output_activation = softmax(hidden_2_output_transfer)

    return (input_2_hidden_transfer, input_2_hidden_activation), (hidden_2_output_transfer, hidden_2_output_activation)


def backward_one_word(word_vector: np.ndarray, context_matrix: np.ndarray,
                      forward_results: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
                      network_hyperparams: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes the word vector it's working on currently, as well as its context.
    Also, the results for all steps in the forward pass and the weights for each layer

    Returns gradients for the network hyperparameters in the order of layers
    """
    input_2_hidden_weights, hidden_2_output_weights = network_hyperparams
    (input_2_hidden_transfer, input_2_hidden_activation),\
        (hidden_2_output_transfer, hidden_2_output_activation) = forward_results

    grad_hidden_2_output_transfer =\
        np.sum(hidden_2_output_activation - context_matrix, axis=0, keepdims=True) / context_matrix.shape[0]
    grad_hidden_2_output_weights = input_2_hidden_activation.T @ grad_hidden_2_output_transfer

    grad_input_2_hidden_activation = grad_hidden_2_output_transfer @ hidden_2_output_weights.T
    grad_input_2_hidden_transfer = grad_input_2_hidden_activation * relu_derivative(input_2_hidden_transfer)
    grad_input_2_hidden_weights = word_vector.T @ grad_input_2_hidden_transfer

    return grad_input_2_hidden_weights, grad_hidden_2_output_weights


def loss_one_word(word_output: np.ndarray, context_matrix: np.ndarray) -> float:
    """
    Calculates log loss for a single word based on its output and context
    """
    loss = np.sum(-1*(context_matrix * np.log(word_output)))
    return loss / context_matrix.shape[0]


def update_hyperparameter(hyperparameter: np.ndarray, grad_hyperparameter: np.ndarray,
                          m: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Does ADAM optimization on hyperparameter

    Returns the newly calculated hyperparameter and optimizer parameters
    """
    new_m = OPTIMIZER["omega1"] * m + (1 - OPTIMIZER["omega1"]) * grad_hyperparameter
    new_v = OPTIMIZER["omega2"] * v + (1 - OPTIMIZER["omega2"]) * (grad_hyperparameter**2)

    m_hat = new_m / (1 - OPTIMIZER["omega1"])
    v_hat = np.abs(new_v) / (1 - OPTIMIZER["omega2"])

    new_hyperparameter = hyperparameter - (LEARN_RATE / (v_hat + OPTIMIZER["norm"])**0.5) * m_hat
    return new_hyperparameter, new_m, new_v


def one_word_iteration(word_vector: np.ndarray, context_matrix: np.ndarray,
                       network_hyperparameters: Tuple[np.ndarray, np.ndarray],
                       ms: Tuple[np.ndarray, np.ndarray], vs: Tuple[np.ndarray, np.ndarray])\
        -> Tuple[Tuple[np.ndarray, np.ndarray], float, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Does forward pass, loss calculation, backward pass and hyperparameter optimization for one word

    Returns the new hyperparameters and loss for that word, as well as the newly calculated optimizer parameters
    """
    forward_results = forward_one_word(word_vector, network_hyperparameters)

    _, (_, hidden_2_output_activation) = forward_results
    loss = loss_one_word(hidden_2_output_activation, context_matrix)

    grad_hyperparameters = backward_one_word(word_vector, context_matrix, forward_results, network_hyperparameters)

    input_2_hidden_updated, hidden_2_output_updated = tuple(
        update_hyperparameter(hyperparameter, grad_hyperparameter, m, v)
        for hyperparameter, grad_hyperparameter, m, v in zip(network_hyperparameters, grad_hyperparameters, ms, vs)
    )

    return (
        (input_2_hidden_updated[0], hidden_2_output_updated[0]),
        loss,
        (input_2_hidden_updated[1], hidden_2_output_updated[1]),
        (input_2_hidden_updated[2], hidden_2_output_updated[2])
    )


def train(word_map: Dict[str, Dict[str, np.ndarray]], plot_graph: bool = False) -> np.ndarray:
    """
    Given a word map with input vectors and their context vectors, embeds the vector space.
    plot_graph - whether to plot the loss graph
    """
    words = [*word_map.keys()]
    hyperparameters = make_network_hyperparameters(word_map[words[0]]["input"].shape[1], VECTOR_SPACE_SIZE)

    total_loss = []
    for i in tqdm.tqdm(range(EPOCHS)):
        epoch_loss = 0.
        ms = vs = (0., 0.)

        np.random.shuffle(words)
        for word in words:
            word_dict = itemgetter(word)(word_map)
            word_vector = word_dict["input"]
            word_context = word_dict["context"]

            hyperparameters, word_loss, ms, vs = one_word_iteration(word_vector, word_context, hyperparameters, ms, vs)
            epoch_loss += word_loss

        total_loss.append(epoch_loss)

    if plot_graph:
        plt.plot(total_loss)
        plt.show()

    return hyperparameters[0]


def main():
    maps = preprocess_corpus(CORPUS_PATH)
    vector_space = train(maps[-1], True)
    print(vector_space)


if __name__ == '__main__':
    main()
