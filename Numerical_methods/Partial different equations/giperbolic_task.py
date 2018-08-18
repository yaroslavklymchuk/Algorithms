import math
import numpy as np
import matplotlib.pyplot as plt


u0 = lambda x: 3*x*(1-x)
du0 = lambda x: 3 - 6*x
d2u0 = -6
u = lambda x,t: u0(x) * np.cos(np.pi * t)
u1 = lambda t: 0
u2 = lambda t: 0
du_dt = lambda x,t: - u0(x) * np.pi * np.sin(np.pi * t)
du_dx = lambda x,t: du0(x) * np.cos(np.pi * t)
d2u_dt = lambda x,t: - u0(x) * np.power(np.pi, 2) * np.cos(np.pi * t)
d2u_dx = lambda t: d2u0 * np.cos(np.pi * t)
F = lambda x, t: d2u_dt(x,t) - d2u_dx(t)

#T?
def Get_X_T(N, t, K, L=1):
    h = L/N
    return [i * h for i in range(1+N)], [k * t for k in range(1+K)]

N = 10
K = 100
thao = 0.1 # tk = 2, must to calculate a period?
h = 1/N

U = [[0 for i in range(N+1)] for j in range(K+1)]

for n in range(N+1):
    U[0][n] = u0(n * h)

print(np.array(U).shape)

for k in range(1, K+1):
    U[k][0] = u1(k * thao)
    U[k][N] = u2(k * thao)

print(np.array(U))

for n in range(1, N):
    U[1][n] = U[0][n]+(pow(thao, 2)/2)*((U[0][n+1]-2*U[0][n]+U[0][n+1])/pow(h, 2)
                                        + F(Get_X_T(N, thao, K)[0][n], 0))




a = [[0 for i in range(N+1)] for i in range(N+1)]
f = [0 for i in range(N+1)]

sigma = 0.1

for k in range(2, K+1):
    a[0][0] = 1
    a[N][N] = 1
    f[0] = U[k][0]
    f[N] = U[k][N]

    for i in range(N):
        a[i][i] = 1 / pow(thao, 2) + 2 * sigma / pow(h, 2)
        a[i][i-1] = - sigma / pow(h, 2)
        a[i][i+1] = -sigma / pow(h, 2)

        f[i] = (2 * U[k-1][i] - U[k-2][i]) / pow(thao, 2) + \
               (1-2*sigma)*(U[k-1][i+1] - 2*U[k-1][i] + U[k-2][i]) / pow(h, 2) + \
               sigma * (U[k-2][i+1] - 2*U[k-2][i]+U[k-2][i-1]) / pow(h, 2) + \
               sigma * F(i*h, k*thao)+(1-2*sigma)*F(i*h,(k-1)*thao)+sigma*F(i*h, (k-2)*thao)


    y = np.linalg.solve(a, f)
    for i in range(1, N):
        U[k][i] = y[i]

print(np.array(U))

f = [0 for i in range(K+1)]
y = [0 for i in range(K+1)]

file = open('results.txt', 'w')
file.write('t' + '  \t' + '||u||(t)' + '    \t' + 'e(t)')
file.write('\n')
for i in range(K+1):
    for j in range(N+1):

        if (math.fabs(U[i][j]) > y[i]):
            y[i] = math.fabs(U[i][j])
        elif (f[i] < math.fabs(u(h*j, thao*i) - U[i][j])):
            f[i] = math.fabs(u(h*j, thao*i) - U[i][j])


        file.write(str(thao*i)+'        \t'+str(y[i]) + '           \t\t'+str(f[i]))
        file.write('\n')

file.write('\n')

file.write('t = %s' % str(2))
file.write('x\t\t' + ' \t\t' + 'u(x)\t\t' + '   \t\t' + 'e(x)')
file.write('\n')
for i in range(N+1):
    file.write(str(h*i)+' \t'+str(U[k][i])+'\t'+str(math.fabs(U[k][i] - u(i*h, 2))))
    file.write('\n')

file.close()


X = [i for i in range(N+1)]
U_ = [U[k][i] for i in range(N+1)]
e_ = [math.fabs(U[k][i] - u(i*h, 2)) for i in range(N+1)]

plt.plot(X, U_, 'r')
plt.show()
plt.plot(X, e_)
plt.show()