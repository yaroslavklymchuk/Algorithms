import numpy as np
import math
import matplotlib.pyplot as plt

'''
(1)
cos(x+a) + b*y = c
x + sin(y+b) = d
'''

'''
(2)
sin(x+y) + cx = d
x^2 + y^2 = 1
'''
'''
a = 0.966
b = -1.493
c = 0.651
d = 0.779
alpha = 0.663
beta = -1.659
'''

class Non_Linear_System:
    def __init__(self, a,b,c,d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    #X = (x1, x2,...,xn).T
    def Represent_matrix(self, x, y):
        Matrix = np.zeros((2,2))
        Matrix[0][0] = (self.c - math.cos(x + self.a)) / self.b
        Matrix[1][0] = self.d - math.sin(Matrix[0][0] + self.b)
        Matrix[0][1] = math.cos(x + self.a) + self.b * y - self.c
        Matrix[1][1] = x + math.sin(y + self.b) - self.d
        return Matrix

    def norm(self, mas, mas1):
        return pow(mas, 2) + pow(mas1, 2)

    def norm1(self, x, x1):
        return sum(((x1[i] - x[i])**2) for i in range(len(x))) ** 0.5

    def Simple_iteration(self, x0,y0, eps):
        prevX = [x0, y0]
        nextX = [self.Represent_matrix(x0, y0)[1][0], self.Represent_matrix(x0, y0)[0][0]]
        number_iter = 0
        norma = 0

        f = open('Results.txt', 'w')
        f.write('\n')
        f.write('Simple_iteration \n')
        f.write('\n')
        f.close()
        while not(self.norm(self.Represent_matrix(prevX[0], prevX[1])[0][1], self.Represent_matrix(prevX[0], prevX[1])[1][1]) < eps and self.norm1(prevX, nextX) < eps):
            prevX = nextX
            nextX = [self.Represent_matrix(prevX[0], prevX[1])[1][0], self.Represent_matrix(prevX[0], prevX[1])[0][0]]

            number_iter += 1
            norma = self.norm(self.Represent_matrix(prevX[0], prevX[1])[0][1], self.Represent_matrix(prevX[0], prevX[1])[1][1])
            print('Решение: х = {}, y = {}'.format(nextX[0], nextX[1]))
            print('Невязка: {}'.format(self.Represent_matrix(nextX[0], nextX[1])[:][1]))
            print('Норма функции: {}'.format(norma))
            print('\n')
            f = open('Results.txt', 'a+')
            f.write("Номер итерации ")
            f.write(str(number_iter))
            f.write('\n')
            #f.write('\n')
            f.write(str('Решение: x = {}, y = {}'.format(nextX[0], nextX[1])))
            f.write('\n')
            #f.write('\n')
            f.write(str('Вектор невязки: {}'.format(self.Represent_matrix(nextX[0], nextX[1])[:][1])))
            f.write('\n')
            #f.write('\n')
            f.write(str('Норма функции: {}'.format(norma)))
            f.write('\n')
            f.close()


        print('\n')
        #return nextX, self.Represent_matrix(nextX[0], nextX[1]), norma


    def Matrix_Yakobi(self, x, y):
        Matrix_der = np.zeros((2,2))
        Matrix_der[0][0] = math.cos(x + y) + self.c
        Matrix_der[0][1] = math.cos(x + y)
        Matrix_der[1][0] = 2*x
        Matrix_der[1][1] = 2*y
        return Matrix_der

    def RepresentMatrixForNewthon(self, x, y):
        Matr = [0, 0]
        Matr[0] = math.sin(x + y) + self.c * x - self.d
        Matr[1] = pow(x, 2) + pow(y, 2) - 1
        return Matr

    def Newthon(self, x0, y0, eps):
        prevX = [x0, y0]
        nextX = np.array(prevX) - np.dot(np.linalg.inv(self.Matrix_Yakobi(prevX[0], prevX[1])), self.RepresentMatrixForNewthon(prevX[0], prevX[1]))
        norma = 0
        number_iter = 0

        f = open('Results.txt', 'a+')
        f.write('\n')
        f.write('Newthon \n')
        f.write('\n')
        while not(self.norm(self.RepresentMatrixForNewthon(prevX[0], prevX[1])[0], self.RepresentMatrixForNewthon(prevX[0], prevX[1])[1]) < eps and self.norm1(prevX, nextX) < eps):
            prevX = nextX
            nextX = np.array(prevX) - np.dot(np.linalg.inv(self.Matrix_Yakobi(prevX[0], prevX[1])), self.RepresentMatrixForNewthon(prevX[0], prevX[1]))
            number_iter += 1
            norma = self.norm(self.RepresentMatrixForNewthon(prevX[0], prevX[1])[0], self.RepresentMatrixForNewthon(prevX[0], prevX[1])[1])
            print('Решение: x = {}, y = {}'.format(nextX[0], nextX[1]))
            print('Вектор невязки: {}'.format(self.RepresentMatrixForNewthon(nextX[0], nextX[1])))
            print('Норма функции: {}'.format(norma))
            print('\n')
            f = open('Results.txt', 'a+')
            f.write("Номер итерации ")
            f.write(str(number_iter))
            f.write('\n')
            #f.write('\n')
            f.write(str('Решение: x = {}, y = {}'.format(nextX[0], nextX[1])))
            f.write('\n')
            #f.write('\n')
            f.write(str('Вектор невязки: {}'.format(self.RepresentMatrixForNewthon(nextX[0], nextX[1]))))
            f.write('\n')
            #f.write('\n')
            f.write(str('Норма функции: {}'.format(norma)))
            f.write('\n')
            #f.write('\n')
            f.close()
        print('\n')
        #return nextX, self.RepresentMatrixForNewthon(nextX[0], nextX[1]), norma


    def plot_graphs(self):
        X = np.linspace(-10, 10, 200)
        Y = np.linspace(-10, 10, 100)
        t = np.arange(0, 2 * np.pi, 0.01)
        r = 1

        plt.subplot(222)  # нарисовали (2) +- неплохо
        plt.plot(r * np.cos(t), r * np.sin(t), 'g', X, np.arcsin(self.d - self.c * X) - X, 'r')
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(r'x^2 + y^2 = 1; sin(x+y) + cx = d')
        plt.axis('equal')

        plt.subplot(223)  # нарисовали (1)
        plt.plot(X, (self.c - np.cos(X + self.a)) / self.b, 'y', X, np.arcsin(self.d - X) - self.b - 0.5 * np.pi, 'b')
        plt.grid(True)
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.title(r'cos(x+a) + b*y = c; x + sin(y+b) = d')
        plt.show()

obj = Non_Linear_System(0.966, -1.493, 0.651, 0.779)

obj.plot_graphs()
print(obj.Simple_iteration(1.5, -1, 1e-5))
print(obj.Newthon(0.7, -0.5, 1e-5))
print(obj.Newthon(0, 1, 1e-5))






