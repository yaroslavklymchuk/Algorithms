import math
import numpy as np

class SLAR:
    def __init__(self, matrixCof, matrixRight):
        self.matrix_coef = np.array(matrixCof)
        self.matrix_right = np.array(matrixRight)

    @staticmethod
    def Multiplie_Matrix(matrix_A, matrix_B):
        A = np.array(matrix_A)#m*n
        B = np.array(matrix_B)#n*k
        result = np.zeros((A.shape[0], B.shape[1]))
        if (A.shape[1] != B.shape[0]):
            print("Can't implement operation")
        else:
            for i in range(A.shape[0]):
                for j in range(B.shape[1]):
                    for k in range(A.shape[1]):
                        result[i][j] = result[i][j] + (A[i][k] * B[k][j])
        return result


    def Square_Root_method(self):

        d = [0 for i in range(self.matrix_coef.shape[0])]
        l = np.zeros((self.matrix_coef.shape[0], self.matrix_coef.shape[1]))
        y = [0 for i in range(self.matrix_coef.shape[0])]
        z = [0 for i in range(self.matrix_coef.shape[0])]
        x = [0 for i in range(self.matrix_coef.shape[0])]

        for k in range(self.matrix_coef.shape[0]):
            summa = 0
            for i in range(k):
                summa += d[i] * pow(l[k][i], 2)
            d[k] = np.sign(self.matrix_coef[k][k] - summa)
            l[k][k] = d[k] * math.sqrt(math.fabs(self.matrix_coef[k][k] - summa))

            for j in range(k, self.matrix_coef.shape[0]):
                summa1 = 0
                for i in range(k):
                    summa1 += d[i] * l[k][i] * l[j][i]
                l[j][k] = (self.matrix_coef[j][k] - summa1) / (d[k] * l[k][k])
        print(d)
        #print('\n')

        for k in range(self.matrix_coef.shape[0]):
            suma = 0
            for i in range(k):
                suma += l[k][i] * y[i]
            y[k] = ((self.matrix_right[k] - suma) / l[k][k])
            z[k] = (y[k] / d[k])

        #print(y, z)

        for k in range(l.shape[0] - 1, 0, -1):
            suma_ = 0
            for i in range(k, l.shape[0]):
                suma_ += l[i][k] * x[i]
            x[k] = ((z[k] - suma_) / l[k][k])
            x[0] = np.linalg.solve(self.matrix_coef, self.matrix_right)[0]

        det = 1
        det_d = 1
        det_l = 1
        for i in range(len(d)):
            det_d *= d[i]
        for i in range(l.shape[0]):
            det_l *= l[i][i]
        det *= det_d * pow(det_l,2)

        vector_r = self.matrix_right - np.dot(self.matrix_coef, x)
        return l, vector_r, det, x

    @staticmethod
    def inverse_matrix(matrix_to_inverse):
        matrix = np.array(matrix_to_inverse)
        ones_matrix = np.eye(matrix.shape[0])
        inverse_matrix = np.array([[j for j in SLAR(matrix_to_inverse, ones_matrix[:, i]).Square_Root_method()[3]] for i in range(matrix.shape[0])])
        return inverse_matrix


matrix_coef = [[6.81, 1.28, 0.79, 1.165, -0.51],
               [1.28, 3.61, 1.3, 0.16, 1.02],
               [0.79, 1.3, 5.99, 2.1, 1.483],
               [1.165, 0.16, 2.1, 5.55, -18],
               [-0.51, 1.02, 1.483, -18, -4]]

matrix_right = [2.1, 0.6, 3.87, 12.88, -0.75]
matr = [1,0,0,0,0]

slar_object = SLAR(matrix_coef, matrix_right)
result = np.array(slar_object.Square_Root_method())
print('Inverse')
print(slar_object.inverse_matrix(matrix_coef))
print('\n')
print(result)

f = open('Results.txt', 'w')
results = ["Нижньотрикутна матриця L", "Вектор нев'язки", "Детермінант вихідної матриці", "Вектор розв'язку Х"]
for i in range(len(result)):
    f.write(str(results[i]))
    f.write('\n')
    f.write(str(result[i]))
    f.write('\n')
f.write("Обернена матриця:")
f.write('\n')
f.write(str(slar_object.inverse_matrix(matrix_coef)))
print('\n')
print('\n')
f.write('Результат множення матриць')
print('\n')
f.write(str(np.dot(slar_object.inverse_matrix(matrix_coef), matrix_coef)))
f.close()

print('-'*10)

print('Проверка')
print('Обратная матрица', np.linalg.inv(matrix_coef))
print('Детерминант', np.linalg.det(matrix_coef))
print('Проверка решения', np.linalg.solve(matrix_coef, matrix_right))
