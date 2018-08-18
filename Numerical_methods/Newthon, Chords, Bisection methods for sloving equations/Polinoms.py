import math


class Polinom:
    def __init__(self, vect):
        self.coef = vect
    def value_pol(self, x):
        polinom = 0
        for i in range(0, len(self.coef)):
            polinom += self.coef[i] * pow(x,i)
        return polinom

    def Bisection(self, left, right, eps):
        number_iter = 0
        x_mean = (left + right) / 2
        interval_result = []

        while ((math.fabs(left - right) > eps) or (math.fabs(self.value_pol(x_mean) > eps))):
            x_mean = (left + right) / 2

            if (self.value_pol(left) * self.value_pol(x_mean) < 0):
                left = left
                right = x_mean
            else:
                left = x_mean
                right = right
            number_iter += 1

            interval_result.append(left)
            interval_result.append(right)
            interval_result.append(self.value_pol(left))
            interval_result.append(self.value_pol(right))

        return [x_mean, number_iter], interval_result



    def derivative(self, x):
        derivative = 0
        for i in range(1, len(self.coef)):
            derivative += self.coef[i] * i * pow(x, i-1)
        return derivative


    def newton(self, left, right, eps):
        x0 = (left + right) / 2
        x1 = x0 - (self.value_pol(x0) / self.derivative(x0))
        number_iter = 0
        inter_result = []
        while (math.fabs(self.value_pol(x1)) > eps or math.fabs(x1 - x0) > eps):
            x0 = x1
            x1 = x0 - (self.value_pol(x0) / self.derivative(x0))
            number_iter += 1
            inter_result.append(x1)
            inter_result.append(self.value_pol(x1))
        return [x1, number_iter], inter_result

    def chords_method(self, left, right, eps):
        val = left
        val1 = right
        x0 = val
        x1 = val1
        number_iter = 0
        inter_result = []
        while (math.fabs(x1 - x0) > eps or math.fabs(self.value_pol(x1)) > eps):

            x0 = x1
            x1 = (val * self.value_pol(val1) - val1 * self.value_pol(val)) / (self.value_pol(val1) - self.value_pol(val))
            if (self.value_pol(val) * self.value_pol(x1) > 0):
                val = x1
            else:
                val1 = x1
            number_iter += 1

            inter_result.append(x0)
            inter_result.append(x1)
            inter_result.append(self.value_pol(x0))
            inter_result.append(self.value_pol(x1))
        return [x1, number_iter], inter_result


pol = Polinom([3, 0, 0, -1, -3, 2])
intervals = [[-1, -0.5], [0.5, 1.4], [1.5, 2.5]]

f = open("Result.txt", 'w')
f.write('Bisection ')
f.write('\n' * 2)
f.close()

f1 = open('Result.txt', 'a+')
for interval in intervals:
    f1.write(str(interval))
    f1.write('\n')
    a = interval[0]
    b = interval[1]

    for item in pol.Bisection(a, b, 1e-6)[0]:
        f1.write("%s " % item)
    f1.write('\n')
    f1.write('\n')

    i = 0
    for items in pol.Bisection(a, b, 1e-6)[1]:
        f1.write('%s ' % items)
        i += 1

        if (i == 4):
            f1.write('\n')
            i = 0
    f1.write('\n')


f1.write('Newton method')
f1.write('\n' * 2)
for interval in intervals:
    f1.write(str(interval))
    f1.write('\n')
    a = interval[0]
    b = interval[1]
    for item in pol.newton(a, b, 1e-6)[0]:
        f1.write("%s " % item)
    f1.write('\n')
    f1.write('\n')

    i = 0
    for items in pol.newton(a, b, 1e-6)[1]:
        f1.write('%s ' % items)
        i += 1
        if (i == 2):
            f1.write('\n')
            i = 0
    f1.write('\n')

f1.write('Chords')
f1.write('\n' * 2)
for interval in intervals:
    f1.write(str(interval))
    f1.write('\n')
    a = interval[0]
    b = interval[1]
    for item in pol.chords_method(a, b, 1e-6)[0]:
        f1.write("%s " % item)
    f1.write('\n')
    f1.write('\n')

    i = 0
    for items in pol.chords_method(a, b, 1e-6)[1]:
        f1.write('%s ' % items)
        i += 1
        if (i == 4):
            f1.write('\n')
            i = 0
    f1.write('\n')
f1.close()
