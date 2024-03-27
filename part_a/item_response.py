from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
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

    for i in range(num_items):
        user_id = data['user_id'][i]
        question_id = data['question_id'][i]
        is_correct = data['is_correct'][i]
        
        if not np.isnan(is_correct):
            p_i = 1 / (1 + np.exp(beta[question_id] - theta[user_id]))
            log_lklihood += is_correct * np.log(p_i) + (1 - is_correct) * np.log(1 - p_i)
            
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
    num_users = len(theta)
    num_questions = len(beta)
    
    for i in range(len(data['user_id'])):
        user_id = data['user_id'][i]
        question_id = data['question_id'][i]
        is_correct = data['is_correct'][i]
        
        if not np.isnan(is_correct):
            p_i = sigmoid(beta[question_id] - theta[user_id])
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

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst


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

    # Define hyperparameters to tune (change values)
    learning_rates = [0.001, 0.01, 0.1]
    iterations = [100, 200, 300]

    best_val_accuracy = 0
    best_test_accuracy = 0
    best_lr = None
    best_iter = None

    for lr in learning_rates:
        for num_iter in iterations:
            # Train the IRT model
            theta, beta, _ = irt(train_data, val_data, lr, num_iter)

            # Evaluate on validation data
            val_accuracy = evaluate(data=val_data, theta=theta, beta=beta)

            # Evaluate on test data
            test_accuracy = evaluate(data=test_data, theta=theta, beta=beta)

            print(f"LR: {lr}, Iterations: {num_iter}, Validation Accuracy: {val_accuracy}, Test Accuracy: {test_accuracy}")

            # Update best accuracy and hyperparameters if needed
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_test_accuracy = test_accuracy
                best_lr = lr
                best_iter = num_iter

    print(f"Best Validation Accuracy: {best_val_accuracy}, Test Accuracy: {best_test_accuracy}, LR: {best_lr}, Iterations: {best_iter}")

    # Plot the training and validation log-likelihoods
    neg_lld_train = [neg_log_likelihood(train_data, theta=best_theta, beta=best_beta) for theta, beta in zip([best_theta]*best_iter, [best_beta]*best_iter)]
    plt.plot(range(1, len(neg_lld_train) + 1), neg_lld_train, label='Training Log-Likelihood')
    plt.plot(range(1, len(best_val_acc_lst) + 1), best_val_acc_lst, label='Validation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Training Log-Likelihood and Validation Accuracy')
    plt.legend()
    plt.show()

    # Evaluate the final model on validation and test datasets
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
