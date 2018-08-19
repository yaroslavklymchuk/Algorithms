#Странно ошибка рисуеться

'''
a = 1.600
b = -0.980
c = -2.069
d = 1.759
e = -0.078

x1 = 1.2
x2 = 1.5

a0 = 0
b0 = 1
a1 = 2
b1 = -1


p(x) = 3
q(x) = -1/x
'''


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

y = lambda x, a, b, c, d, e: a * pow(x, 2) + b * x + c + 1 / (d * x + e)

dy = lambda x, a, b, c, d, e: 2 * a * x + b - d / (pow(d * x + e, 2))

d2y = lambda x, a, b, c, d, e: 2 * a + (pow(d, 2) * 2) / (pow(d * x + e, 3))

p = lambda x: 3
q = lambda x: -1/x

class Boundary_Task:
    def __init__(self, a, b, c, d, e, x1, x2, a0, b0, a1, b1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.x1 = x1
        self.x2 = x2
        self.a0 = a0
        self.b0 = b0
        self.a1 = a1
        self.b1 = b1


    @staticmethod
    def a1y(self):
        return self.a0 * y(self.x1, self.a, self.b, self.c, self.d, self.e) + \
               self.b0 * dy(self.x1, self.a, self.b, self.c, self.d, self.e)

    @staticmethod
    def a2y(self):
        return self.a1 * y(self.x2, self.a, self.b, self.c, self.d, self.e) + \
               self.b1 * dy(self.x2, self.a, self.b, self.c, self.d, self.e)

    def A(self):
        return self.a0 * y(self.x1, self.a, self.b, self.c, self.d, self.e) + \
               self.b0 * Boundary_Task.a1y(self)

    def B(self):
        return self.a1 * y(self.x2, self.a, self.b, self.c, self.d, self.e) + \
               self.b1 * dy(self.x2, self.a, self.b, self.c, self.d, self.e)


    f = lambda self, x: d2y(x, self.a, self.b, self.c, self.d, self.e) + \
                        p(x) * dy(x, self.a, self.b, self.c, self.d, self.e) + \
                        q(x) * y(x, self.a, self.b, self.c, self.d, self.e)


def error(y_true, y_pred):
    return [(el - el1) for el, el1 in zip(y_true, y_pred)]

def error_(y_true, y_pred, h):
    return np.sqrt(np.sum([np.power(el, 2) for el in error(y_true, y_pred)]) * h)

def Get_X(x1, x2, N):
    h = (x2 - x1) / N

    X = [(x1 + i*h) for i in range(N+1)]
    Y0 = [0 for i in range(N+1)]

    return X, Y0, h

def process(obj, N):

    m = [0 for i in range(N)]
    n = [0 for i in range(N)]
    c = [0 for i in range(N)]
    d = [0 for i in range(N)]

    h = Get_X(obj.x1, obj.x2, N)[2]
    X = Get_X(obj.x1, obj.x2, N)[0]

    m[0] = h * p(X[0]) - 2
    n[0] = 1 - h*p(X[0])+pow(h, 2)*q(X[0])
    c[0] = (obj.b0 - obj.a0*h) / (m[0]*(obj.b0-obj.a0*h) + n[0]*obj.b0)
    d[0] = n[0]*obj.A()*h/(obj.b0-obj.a0*h)+obj.f(X[0])*pow(h, 2)

    print('{}, {}, {}, {}'.format(m[0], n[0], c[0], d[0]))

    for i in range(1, N-1):
        m[i] = h * p(X[i]) - 2
        n[i] = 1 - h * p(X[i]) + pow(h, 2) * q(X[i])
        c[i] = 1/(m[i] - n[i]*c[i-1])
        d[i] = obj.f(X[i])*pow(h, 2) - n[i]*c[i-1]*d[i-1]

    print('\n')
    Y0 = Get_X(obj.x1, obj.x2, N)[1]
    Y0[N] = (obj.b1*c[N-2]*d[N-2]+obj.B()*h) / (obj.b1*(1+c[N-2])+obj.a1*h)

    for i in range(N-1, 0, -1):
        Y0[i] = c[i-1]*(d[i-1]-Y0[i+1])

    Y0[0] = (obj.b0*Y0[1]-obj.A()*h)/(obj.b0-obj.a0*h)


    for i in range(N+1):
        print('{}, x={}, {} vs {}, {}'.format(i, X[i], y(X[i], obj.a, obj.b, obj.c, obj.d, obj.e),
                                    Y0[i], (y(X[i], obj.a, obj.b, obj.c, obj.d, obj.e) - Y0[i])))

    frame = pd.DataFrame(columns=['Y_real', 'Y_prediction', 'Error'])
    frame['Y_real'] = pd.Series(np.array([y(X[i], obj.a, obj.b, obj.c, obj.d, obj.e) for i in range(N+1)]))
    frame['Y_prediction'] = pd.Series(np.array([Y0[i] for i in range(N+1)]))
    frame['Error'] = pd.Series(np.array(error(frame['Y_real'].values, frame['Y_prediction'].values)))
    frame.to_csv('results.csv', index = False)

    return Y0

N = 10000

obj = Boundary_Task(1.600,-0.980, -2.069, 1.759,-0.078, 1.2, 1.5, 0, 1, 2, -1)

H = np.linspace(0.0001, 0.1, 100)
answer = [y(Get_X(obj.x1, obj.x2, N)[0][i], obj.a, obj.b, obj.c, obj.d, obj.e) for i in range(N+1)]
prediction = process(obj,N)

error_values = [error_(answer, prediction, h) for h in H]

plt.plot(H, error_values, 'r')
plt.title('График зависимости ошибки(L2) от h')
plt.xlabel('H')
plt.ylabel('Error')
plt.show()