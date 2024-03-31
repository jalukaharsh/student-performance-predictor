from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.
    num_items = len(data['user_id'])
    epsilon = 1e-15

    for i in range(num_items):
        user_id = data['user_id'][i]
        question_id = data['question_id'][i]
        is_correct = data['is_correct'][i]

        # Check if is_correct is not NaN
        if not np.isnan(is_correct):
            # Compute the logit
            logit = beta[question_id] - theta[user_id]

            # Compute the sigmoid function with logit
            p_i = sigmoid(logit)

            # Compute the log-likelihood contribution
            log_lklihood += is_correct * np.log(p_i + epsilon) + (1 - is_correct) * np.log(1 - p_i + epsilon)

    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    num_items = len(data['user_id'])
    epsilon = 1e-15

    for i in range(num_items):
        user_id = data['user_id'][i]
        question_id = data['question_id'][i]
        is_correct = data['is_correct'][i]

        if not np.isnan(is_correct):
            # Compute the logit
            logit = beta[question_id] - theta[user_id]

            # Compute the sigmoid function with logit
            p_i = sigmoid(logit)

            # Compute the gradients
            grad_theta = (is_correct - p_i) * beta[question_id]
            grad_beta = (is_correct - p_i) * (theta[user_id] - beta[question_id])

            # Update theta and beta
            theta[user_id] += lr * grad_theta
            beta[question_id] += lr * grad_beta

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    num_users = len(set(data['user_id']))
    num_questions = len(set(data['question_id']))
    theta = np.random.rand(num_users)
    beta = np.random.rand(num_questions)

    val_acc_lst = []
    train_nlld_lst = []
    val_nlld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_nlld_lst.append(neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
        neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_nlld_lst.append(neg_lld)

    return theta, beta, val_acc_lst, train_nlld_lst, val_nlld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Hyperparameters
    learning_rates = [0.0001, 0.001, 0.01]
    iterations = [25, 35, 45]

    # Tune the hyperparamaters
    best_val_accuracy = 0
    best_lr = None
    best_iter = None
    best_theta = None
    best_beta = None

    for lr in learning_rates:
        for num_iter in iterations:
            theta, beta, val_acc_lst, train_nlld_lst, val_nlld_lst = irt(train_data, val_data, lr, num_iter)
            val_accuracy = evaluate(data=val_data, theta=theta, beta=beta)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_lr = lr
                best_iter = num_iter
                best_theta = theta
                best_beta = beta

    print(f"Best Learning Rate: {best_lr}, Best Number of Iterations: {best_iter}")

    # Plot the training and validation log-likelihoods as a function of iteration
    theta, beta, val_acc_lst, train_nlld_lst, val_nlld_lst = irt(train_data, val_data, best_lr, best_iter)

    plt.plot(range(1, best_iter + 1), train_nlld_lst, label='Training Log-Likelihood')
    plt.plot(range(1, best_iter + 1), val_nlld_lst, label='Validation Log-Likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Training and Validation Log-Likelihoods')
    plt.legend()
    plt.show()

    # Report the validation and test accuracies of the final model
    val_accuracy = evaluate(data=val_data, theta=best_theta, beta=best_beta)
    test_accuracy = evaluate(data=test_data, theta=best_theta, beta=best_beta)

    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
