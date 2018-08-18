import numpy as np
import math, random


matrix = np.array([[7.14, 1.28, 0.79, 1.12],
                  [1.28, 3.28, 1.3, 0.16],
                  [0.79, 1.3, 6.32, 2.1],
                  [1.12, 0.16, 2.1, 5.22]])


def Degree_method(matrix, eps):
    max_iter = 100
    X = [i for i in range(max_iter)]
    X[0] = np.array([i for i in range(matrix.shape[0])]).T
    X[1] = np.dot(matrix, X[0])
    i = random.choice([i for i in range(len(X[1]))])
    lambda_1 = (X[1][i]) / (X[0][i])
    lambda_0 = 1
    k = 1
    while (np.fabs(lambda_1 - lambda_0) > eps):
        X[k] = np.dot(matrix, X[k - 1])
        X[k + 1] = np.dot(matrix, X[k])
        i = random.choice([i for i in range(len(X[k]))])
        lambda_0 = (X[k][i]) / (X[k - 1][i])
        lambda_1 = X[k + 1][i] / X[k][i]

        k += 1
    return lambda_1, X[k] / np.linalg.norm(X[k])

'''
print('Max eigenvalue by degree method: {}'.format(Degree_method(matrix, 1e-5)[0]))
print('Eigenvector: {}'.format(Degree_method(matrix, 1e-5)[1]))

#print('Min eigenvalue by degree method: {}'\
      .format(Degree_method(matrix -\
np.dot(Degree_method(matrix, 1e-5)[0], np.eye(4)), 1e-5)[0] + \
              Degree_method(matrix, 1e-5)[0]))

#print('Eigenvector: {}'.format(Degree_method(matrix - \
                    np.dot(Degree_method(matrix, 1e-5)[0], np.eye(4)), 1e-5)[1]))
'''

with open ('file_degree.txt', 'w') as file:
    file.write('Max eigenvalue by degree method: {}'.format(Degree_method(matrix, 1e-5)[0]))
    file.write('\n')
    file.write('Eigenvector: {}'.format(Degree_method(matrix, 1e-5)[1]))
    file.write('\n')

    file.write('Min eigenvalue by degree method: {}' \
          .format(Degree_method(matrix - \
            np.dot(Degree_method(matrix, 1e-5)[0], np.eye(4)), 1e-5)[0] + \
                  Degree_method(matrix, 1e-5)[0]))
    file.write('\n')
    file.write('Eigenvector: {}'.format(Degree_method(matrix - \
         np.dot(Degree_method(matrix, 1e-5)[0], np.eye(4)), 1e-5)[1]))

def find_index(matrix):
    max_ = np.fabs(matrix[0][1])
    k, t = 0, 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i !=j and i < j) and np.fabs(matrix[i][j]) > max_):
                max_ = np.fabs(matrix[i][j])
                k, t = i, j
    return k, t


find_t = lambda x: 1/(x - np.sqrt(pow(x,2) + 1))\
    if x < 0 else 1/(x + np.sqrt(pow(x,2) + 1))

def make_T(I, c, s, i, j):
    I[i][i], I[j][j] = c, c
    if (j > i):
        I[i][j] = s
        I[j][i] = -s
    else:
        I[i][j] = -s
        I[j][i] = s
    return I

def W_2(matrix):
    return pow(np.linalg.norm(matrix, 'fro'), 2) - \
sum([pow(matrix[i][j], 2) for i in range(matrix.shape[0]) for j \
     in range(matrix.shape[1]) if i == j])


def get_diag_el(matrix):
    return np.array([matrix[i][j] for i in range(matrix.shape[0]) \
                     for j in range(matrix.shape[1]) if i == j])


#some problems with
def jakobi(matrix, eps):
    A = matrix
    iter_ = 0
    Q = np.eye(matrix.shape[0])

    while ((W_2(A)) > eps):
        print("Current matrix: \n{}".format(A))

        i, j = find_index(A)

        print('i, j: {}, {}'.format(i, j))

        alpha, beta, gamma = A[i][i], A[j][j], A[i][j]
        print('alpha, beta, gamma: {}, {}, {}'.format(alpha, beta, gamma))

        ksi = - (alpha - beta) / (2 * gamma)

        c = 1 / np.sqrt(1 + pow(find_t(ksi), 2))
        s = c * find_t(ksi)
        delta = sum([pow(A[i][j], 2) for i in range(A.shape[0]) for j in range(A.shape[1]) if i == j])

        print('ksi, t, c, s: {}, {}, {}, {}'.format(ksi, find_t(ksi), c, s))

        T = make_T(np.eye(A.shape[0]), c, s, i, j)
        Q = np.dot(Q, T)
        print(Q, '\n')
        print(T)

        print(pow(c, 2) + pow(s, 2) == 1)
        print('delta, 2w, frobenius_norm, {}, {}, {}'.format(delta, W_2(A), pow(np.linalg.norm(A, 'fro'), 2)))

        iter_ += 1

        A = np.dot(np.dot(T.T, A), T)
        print('New_matrix: \n{}'.format(A))

    print('nb_iterations: {}'.format(iter_))
    return A, Q


eigenvals = get_diag_el(jakobi(matrix, 1e-5)[0])
eigenvectors = [jakobi(matrix, 1e-5)[1][:, i] for i in range(4)]


print('Residual vector: \n')

for l, v in zip(eigenvals, eigenvectors):
    print(np.dot(matrix, v) - np.dot(l, v))
