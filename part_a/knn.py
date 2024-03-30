from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    transposed_matrix = matrix.T

    # Create the KNN imputer that works on the rows
    nbrs = KNNImputer(n_neighbors=k)
    imputed_matrix = nbrs.fit_transform(transposed_matrix)

    # Transpose the matrix back to the original user-item format
    imputed_matrix = imputed_matrix.T

    # Calculate the accuracy on the validation data
    acc = sparse_matrix_evaluate(valid_data, imputed_matrix)
    print("Validation Accuracy: {}".format(acc))
    return acc



def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    # Define the range of k values to test.
    k_values = [1, 6, 11, 16, 21, 26]
    validation_accuracies = []

    # Compute the validation accuracy for each k.
    for k in k_values:
        print(f"Running k-NN with k = {k}")
        accuracy = knn_impute_by_user(sparse_matrix, val_data, k)
        validation_accuracies.append(accuracy)
        print(f"Validation accuracy for k = {k}: {accuracy}")

    # Plot validation accuracies.
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, validation_accuracies, marker='o')
    plt.title('Validation Accuracy for Different k Values')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    # Pick the best performing k value based on validation accuracy.
    best_k_index = validation_accuracies.index(max(validation_accuracies))
    best_k = k_values[best_k_index]
    print(f"Best k value: {best_k}")

    # Report the test accuracy with the chosen k.
    test_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print(f"Test accuracy with k = {best_k}: {test_accuracy}")

if __name__ == "__main__":
    main()
