import math
import numpy as np
from scipy.misc import derivative
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from sympy import *

'''
y = ((1+x^8)^(3/2))/(12x^12)
'''

'''
 sqrt(x-1)*(3x+2)/(4x^2)
'''

polinom = lambda x: (pow((1 + pow(x, 8)), (3/2)) / (12 * pow(x, 12)))


def Show_6th_deriv():
    x = Symbol('x')
    y = (pow((1 + pow(x, 8)), (3/2)) / (12 * pow(x, 12)))
    der_6_y = (((((y.diff(x)).diff(x)).diff(x)).diff(x)).diff(x)).diff(x)
    print('Исходная функция: ', y)
    print('Производная 6-го порядка: ', der_6_y)



def Get_sixth_deriv(polinom, x):
    return derivative(polinom, x, n = 6, order = 9, dx=1e-5)

def Get_1st_deriv(polinom, x):
    return derivative(polinom, x, dx=1e-5)

def Get_2st_deriv(polinom, x):
    return derivative(polinom,x, n=2, order=5, dx=1e-5)

def product(x, list_x, n):
    mul = 1
    for i in range(len(list_x)):
        if i:
            mul *= x - list_x[i - 1]
        yield mul

def Newthon_forward(list_x, list_y, x):
    C = []

    forward_newthon_polinom = 0

    for n in range(len(list_x)):
        p = product(list_x[n], list_x, n + 1)
        C.append((list_y[n] - sum(C[k] * next(p) for k in range(n))) / next(p))


    forward_newthon_polinom = sum(C[k] * p for k, p in enumerate(product(x,list_x, len(C))))

    return forward_newthon_polinom

def Newthon_reverse(list_x, list_y, x):
    C = []

    reverse_newthon_polinom = 0

    for n in range(len(list_x)):
        p = product(sorted(list_x, reverse=True)[n], sorted(list_x, reverse=True), n + 1)
        C.append((sorted(list_y, reverse=True)[n] - sum(C[k] * next(p) for k in range(n))) / next(p))

    reverse_newthon_polinom = sum(C[k] * p for k, p in enumerate(product(x, sorted(list_x, reverse=True), len(C))))

    return reverse_newthon_polinom

def Lagrange_forward(list_x, list_y, x):

    forward_lagrange_polinom = 0
    for j in range(len(list_y)):
        product = 1 #числитель
        product1 = 1 # знаменатель
        for i in range(len(list_x)):
            if i==j:
                product *= 1
                product1 *= 1
            else:
                product *= (x - list_x[i])
                product1 *= (list_x[j] - list_x[i])
        forward_lagrange_polinom += list_y[j] * (product / product1)
    return forward_lagrange_polinom

def Lagrange_reverse(list_x, list_y, x):

    reverse_lagrange_polinom = 0
    for j in range(len(list_y) - 1, -1, -1):
        product = 1 #числитель
        product1 = 1 # знаменатель
        for i in range(len(list_x) - 1, -1, -1):
            if i==j:
                product *= 1
                product1 *= 1
            else:
                product *= (x - list_x[i])
                product1 *= (list_x[j] - list_x[i])
        reverse_lagrange_polinom += list_y[j] * (product / product1)
    return reverse_lagrange_polinom


def Cubic_Spline(x_list, y_list, x_dop_l, x_dop_r, x):
    N = len(x_list)
    matrix = [[0 for j in range(3 * N)] for i in range(3 * N)]
    rightVector = [0 for i in range(3 * N)]
    for line in range(N - 1):
        matrix[line][line + 1] = 1
        matrix[line][N + line + 1] = -(x_list[line + 1] - x_list[line]) / 2
        matrix[line][2 * N + line + 1] = ((x_list[line + 1] - x_list[line]) ** 2) / 6
        rightVector[line] = (y_list[line + 1] - y_list[line]) / (x_list[line + 1] - x_list[line])
        matrix[N - 1 + line][line + 1] = 1
        matrix[N - 1 + line][line] = -1
        matrix[N - 1 + line][N + line + 1] = -(x_list[line + 1] - x_list[line])
        matrix[N - 1 + line][2 * N + line + 1] = ((x_list[line + 1] - x_list[line]) ** 2) / 2
        matrix[N * 2 - 2 + line][N + line + 1] = 1
        matrix[N * 2 - 2 + line][N + line] = -1
        matrix[N * 2 - 2 + line][2 * N + line + 1] = -(x_list[line + 1] - x_list[line + 1])
    matrix[3 * N - 3][0] = x_list[1] - x_list[0]
    matrix[3 * N - 3][N] = ((x_list[1] - x_list[0]) ** 2) / 2
    matrix[3 * N - 3][2 * N] = -((x_list[1] - x_list[0]) ** 2) / 6
    rightVector[3 * N - 3] = y_list[1] - y_list[0]
    matrix[3 * N - 2][2 * N - 1] = 1
    rightVector[3 * N - 2] = x_dop_r
    matrix[3 * N - 1][N] = 1
    matrix[3 * N - 1][2 * N] = -(x_list[1] - x_list[0])
    rightVector[3 * N - 1] = x_dop_l

    matrix_coef = np.linalg.solve(matrix, rightVector)

    n = len(x_list)
    i = 1
    while not (x_list[i - 1] <= x and x <= x_list[i]): i += 1
    return y_list[i] + matrix_coef[i] * (x - x_list[i]) + matrix_coef[n + i] * (((x - x_list[i])) ** 2) / 2 + matrix_coef[2 * n + i] * ((x - x_list[i]) ** 3) / 6

def Accuracy(x, model_res):
    return math.fabs(polinom(x) - model_res)

def Plot_func_and_6th_deriv():
    x = np.arange(-10, 10, 0.5)
    y = pow((1 + pow(x, 8)), (3/2)) / (12 * pow(x, 12))
    sixth_der = 107520.0*x**30*(x**8 + 1)**(-4.5) - 126720.0*x**22*(x**8 + 1)**(-3.5) + 66240.0*x**14*(x**8 + 1)**(-2.5) + 42000.0*x**6*(x**8 + 1)**(-1.5) + 274680.0*(x**8 + 1)**(-0.5)/x**2 - 1106280.0*(x**8 + 1)**0.5/x**10 + 742560*(x**8 + 1)**1.5/x**18

    plt.subplot(221)
    plt.plot(x, y, 'r')
    plt.title('Function')
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(222)
    plt.plot(x, sixth_der, 'b')
    plt.title('Sixth derivative')
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()

def Plot_Lagrange_results():
    x = [i for i in np.linspace(1, 20, 6)]
    y = [polinom(i) for i in x]
    NewX = [i for i in np.linspace(np.min(x), np.max(x), 100)]
    NewY_forw = [Lagrange_forward(x, y, i) for i in NewX]
    NewY_rev = [Lagrange_reverse(x, y, i) for i in NewX]
    Y = [polinom(i) for i in NewX]#[np.interp(i, x, y) for i in NewX]

    plt.subplot(221)
    plt.plot(x, y, 'o', NewX, NewY_forw)
    plt.title("Forward Lagrange")
    plt.legend(['Узловые значения',"Forward Lagrange"])
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(222)
    plt.plot(x, y, 'o', NewX, NewY_rev)
    plt.title("Reverse Lagrange")
    plt.legend(['Узловые значения',"Reverse Lagrange"])
    plt.xlabel("X")
    plt.ylabel("Y")


    plt.subplot(223)
    plt.plot(x, y, 'o', NewX, Y)
    #plt.title("График исходной функции на этом отрезке")
    plt.legend(['Узловые значения', "Исходная функция"])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig('Метод Лагранжа.png')
    plt.show()

def Plot_Newthon_results():
    x = [i for i in np.linspace(1, 20, 6)]
    y = [polinom(i) for i in x]
    NewX = [i for i in np.linspace(np.min(x), np.max(x), 100)]
    NewY_forw = [Newthon_forward(x, y, i) for i in NewX]
    NewY_rev = [Newthon_reverse(x, y, i) for i in NewX]
    Y = [polinom(i) for i in NewX]

    plt.subplot(221)
    plt.plot(x, y, 'o', NewX, NewY_forw)
    plt.title("Forward Newthon")
    plt.legend(['Узловые значения', "Forward Newthon"])
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(222)
    plt.plot(x, y, 'o', NewX, list(reversed(NewY_rev)))
    plt.title("Reverse Newthon")
    plt.legend(['Узловые значения', "Reverse Newthon"])
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(223)
    plt.plot(x, y, 'o', NewX, Y)
    # plt.title("График исходной функции на этом отрезке")
    plt.legend(['Узловые значения', "Исходная функция"])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig('Метод Ньютона.png')
    plt.show()

def Plot_CubicSpline_results():
    x = [i for i in np.linspace(1, 20, 6)]
    y = [polinom(i) for i in x]
    NewX = [i for i in np.linspace(np.min(x), np.max(x), 100)]
    cs = CubicSpline(x, y)
    NewY = [cs(i) for i in NewX]
    Y = [Cubic_Spline(x, y, 0, 0, i) for i in NewX]
    Y_ = [polinom(i) for i in NewX]

    plt.subplot(221)
    plt.plot(x, y, 'o', NewX, NewY)
    plt.title("Numpy Cubic_Spline")
    plt.legend(['Узловые значения', "Numpy Cubic_spline"])
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(222)
    plt.plot(x, y, 'o', NewX, Y)
    plt.title("CubicSpline")
    plt.legend(['Узловые значения', "Cubic_spline"])
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(223)
    plt.plot(x, y, 'o', NewX, Y_)
    # plt.title("График исходной функции на этом отрезке")
    plt.legend(['Узловые значения', "Исходная функция"])
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.savefig("Кубический сплайн.png")
    plt.show()

def Mod(x, list_x):
    mul = 1
    for i in range(len(list_x)):
        mul *= (x - list_x[i])

    return mul

def Plot_accuracy_graphs():
    x = [i for i in np.linspace(1, 20, 6)]
    y = [polinom(i) for i in x]
    x_test = [i for i in np.linspace(1, 20, 100)]
    cs = CubicSpline(x, y)
    Y_lagr_forw = np.array([Accuracy(i, Lagrange_forward(x, y, i)) for i in x_test])
    Y_lagr_rev = np.array([Accuracy(i, Lagrange_reverse(x, y, i)) for i in x_test])
    Y_newth_forw = np.array([Accuracy(i, Newthon_forward(x, y, i)) for i in x_test])
    Y_newth_rev = np.array([Accuracy(i, Newthon_reverse(x, y, i)) for i in x_test])
    #Y_ = np.array([Accuracy(i, np.interp(i, x, y)) for i in x_test])
    Y_spline = np.array([Accuracy(i, Cubic_Spline(x, y, Get_2st_deriv(polinom, 1), Get_2st_deriv(polinom, 20), i)) for i in x_test])

    modul_6th_der = [math.fabs(Get_sixth_deriv(polinom, i)) for i in x]

    Y_majoarnt = list(map(lambda z: math.fabs(z) / math.factorial(len(x) + 1), [max(modul_6th_der) * Mod(i, x) for i in x_test]))

    plt.title("Графики ошибок")
    plt.plot(x_test, Y_lagr_forw, 'g', x_test, Y_newth_forw, 'r', x_test, Y_newth_forw, 'c',x_test, Y_spline, 'm')#, x_test, Y_majoarnt, 'b')
    plt.legend(["Lagrange_forward", 'Newthon forward', 'Newthon reverse', 'Cubic Spline'])#, 'Мажоранта помилки'])
    plt.savefig('Accuracy.png')
    plt.show()


def Plot_Majorant(list_x):
    x_ = [i for i in np.linspace(1, 20, 6)]
    modul_6th_der = [math.fabs(Get_sixth_deriv(polinom, i)) for i in x_]

    Y = list(map(lambda x: math.fabs(x) / math.factorial(len(x_) + 1), [max(modul_6th_der) * Mod(i, x_) for i in list_x]))
    #print(Y)

    plt.plot(list_x, Y, 'r')
    plt.legend(['Мажоранта ошибки'])
    plt.savefig('Мажоранта ошибки')
    plt.show()

def ShowFunc():

    x = [i for i in np.linspace(1, 20, 6)]
    y = [polinom(i) for i in x]
    print('Forward Lagrange \n', Lagrange_forward(x, y, 5))
    print('Reverse Lagrange \n', Lagrange_reverse(x, y, 5))
    print("Forward Newthon \n", Newthon_forward(x, y, 5))
    print('Reverse Newthon \n', Newthon_reverse(x, y, 5))
    print("Cubic Spline \n", Cubic_Spline(x, y, 0, 0, 5))
    print('Проверка интерполяцией полиномом: ', np.interp(5, x, y))
    cs = CubicSpline(x, y)# библиотечный сплайн 3-й степени
    print('Проверка интерполяцией кубическим сплайном: ', cs(5))

    #выводим значения ошибки на "учебном" отрезке для каждого из методов из шагом в 5 раз меньше чем шаг интерполяции
    x_test = [i for i in np.linspace(1, 20, 50)]
    f = open('Results.txt', 'w')
    f.write('Lagrange: \n')
    f.write(str(np.mean([Accuracy(i, Lagrange_forward(x, y, i)) for i in x_test])))
    f.write('\n')
    f.write('Newthon Forward: \n')
    f.write(str(np.mean([Accuracy(i, Newthon_forward(x, y, i)) for i in x_test])))
    f.write('\n')
    f.write('Newthon reversed: \n')
    f.write(str(np.mean([Accuracy(i, Newthon_reverse(x, y, i)) for i in x_test])))
    f.write('\n')
    f.write('Cubic_Spline: \n')
    f.write(str(np.mean([Accuracy(i, Cubic_Spline(x, y, 0, 0, i)) for i in x_test])))
    f.write('\n')
    f.close()
    print([Accuracy(i, Lagrange_forward(x, y, i)) for i in x_test])
    print([Accuracy(i, Lagrange_reverse(x, y, i)) for i in x_test])
    print([Accuracy(i, Newthon_forward(x, y, i)) for i in x_test])
    print([Accuracy(i, Newthon_reverse(x, y, i)) for i in x_test])
    print([Accuracy(i, Cubic_Spline(x, y, 0, 0, i)) for i in x_test])
    #print([Accuracy(i, cs(i)) for i in x_test])
    #print([Accuracy(i, np.interp(i, x, y)) for i in x_test])

def Plot_together():
    x = np.linspace(1, 20, 1000)
    x_ = np.linspace(1, 20, 6)
    y_ = [polinom(i) for i in x_]
    y = pow((1 + pow(x, 8)), (3/2)) / (12 * pow(x, 12))
    y_newthon = [Newthon_forward(x_, y_, i) for i in x]
    y_lagrange = [Lagrange_forward(x_, y_, i) for i in x]
    y_newthon_reversed = [Newthon_reverse(x_, y_, i) for i in x]
    y_spline = [Cubic_Spline(x_, y_, 0, 0, i) for i in x]

    plt.plot(x_, y_, 'o', x, y,'r', x, y_spline, 'b', x, y_newthon, 'y', x, list(reversed(y_newthon_reversed)), 'c', x, y_lagrange, 'g')
    plt.legend(["Узловые значения", "Исходная функция","Сплайн", "Ньютон", "Обратный Ньютон", "Лагранж"])
    plt.savefig('Результаты.png')
    plt.show()

Plot_together()

Plot_Lagrange_results()
Plot_Newthon_results()
Plot_CubicSpline_results()
Plot_func_and_6th_deriv()

Plot_accuracy_graphs()

x_ = [i for i in np.linspace(1, 20, 100)]
Plot_Majorant(x_)
#ShowFunc()
print('\n')
#Show_6th_deriv()