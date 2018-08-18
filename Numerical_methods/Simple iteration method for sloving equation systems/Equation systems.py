import math
import numpy as np

class SLAR:
    def __init__(self, matrixCof, matrixRight):
        self.matrix_coef = np.array(matrixCof)
        self.matrix_right = np.array(matrixRight)

    @staticmethod
    def Multiplie_Matrix(matrix_A, matrix_B):
        A = np.array(matrix_A)  # m*n
        B = np.array(matrix_B)  # n*k
        result = np.zeros((A.shape[0], B.shape[1]))
        if (A.shape[1] != B.shape[0]):
            print("Can't implement operation")
        else:
            for i in range(A.shape[0]):
                for j in range(B.shape[1]):
                    for k in range(A.shape[1]):
                        result[i][j] = result[i][j] + (A[i][k] * B[k][j])
        return result


    def Simple_iteration_method(self):

        beta = []
        b = [[0 for i in range(self.matrix_coef.shape[0])] for j in range(self.matrix_coef.shape[0])]
        eps = 1e-5
        '''
        print(self.matrix_coef)
        print('-'*30)'''

        for i in range(self.matrix_coef.shape[0]):
            new_matrix = self.matrix_coef[i][self.matrix_coef.shape[0] - 2]
            self.matrix_coef[i][self.matrix_coef.shape[0] - 2] = self.matrix_coef[i][self.matrix_coef.shape[0] - 1]
            self.matrix_coef[i][self.matrix_coef.shape[0] - 1] = new_matrix
        '''
        print (self.matrix_coef)
        print('-'*20)
        print(self.matrix_right)'''

        for i in range(self.matrix_coef.shape[0]):
            beta.append(self.matrix_right[i] / self.matrix_coef[i][i])
            for j in range(self.matrix_coef.shape[0]):
                if (i != j):
                    b[i][j] = -(self.matrix_coef[i][j] / self.matrix_coef[i][i])
                else:
                    b[i][j] = 0

        x_appr = beta
        x = beta + np.dot(b, x_appr)
        number_iter = 0
        f = open('Result.txt', 'w')
        #np.fabs(self.matrix_right - np.dot(self.matrix_coef,x))
        while(np.max(np.linalg.norm((self.matrix_right - np.dot(self.matrix_coef,x)))) > eps):
            number_iter += 1
            x_appr = x
            x = beta + np.dot(b, x_appr)
            print(number_iter)
            print(x)
            print('-'*20)
            print(self.matrix_right - np.dot(self.matrix_coef, x))
            print(np.max(np.fabs(self.matrix_right - np.dot(self.matrix_coef,x))))
        print('\n')
        return x

matrix_coef = [[6.81, 1.28, 0.79, 1.165, -0.51],
               [1.28, 3.61, 1.3, 0.16, 1.02],
               [0.79, 1.3, 5.99, 2.1, 1.483],
               [1.165, 0.16, 2.1, 5.55, -18],
               [-0.51, 1.02, 1.483, -18, -4]]

matrix_right = [2.1, 0.6, 3.87, 12.88, -0.75]

slar_object = SLAR(matrix_coef, matrix_right)
slar = SLAR(matrix_coef, matrix_right)

print(slar.Simple_iteration_method())
print('Проверка')
print(np.linalg.solve(matrix_coef, matrix_right))


