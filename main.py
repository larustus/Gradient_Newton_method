import copy
import time

import numpy as np
import math
from scipy.optimize import rosen
import matplotlib.pyplot as plt


def najszybszy_spadek(p, beta_p):
    start = time.time()
    # p -- poczatkowe punkt,

    eps = 10 ** (-12)  # dokladnosc przyblizenia
    iteracje = 0  # liczba iteracji
    v = p  # pierwsze przyblizenie

    # funkcja do minimalizacji
    # f = (1-x)^2 + 100*(y-x^2)^2

    # wyliczamy gradient tej funkcji
    # grad(f) = [-2*(1-x) - 400*(y-x)*x, 200*(y-x^2)]

    # v - to wczesniejsze przyblizenie,
    # w- to aktualne przyblizenie
    w = copy.deepcopy(v)
    while ((1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2 > eps):  # dopoki nie osiagniemy oczekiwanej dokladnosci
        iteracje = 1
        beta = beta_p
        gradient = [-2 * (1 - v[0]) - 400 * (v[1] - v[0] ** 2) * v[0], 200 * (v[1] - v[0] ** 2)]
        w = [v[0] - beta * gradient[0], v[1] - beta * gradient[1]]  # glowne rownanie metody najmniejszego spadku

        # w przypadku gdy metoda 'przestrzeli'
        while ((1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2 > (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2):
            beta = beta / 2.0
            w = [v[0] - beta * gradient[0], v[1] - beta * gradient[1]]  # glowne rownanie metody najmniejszego spadku
            iteracje += 1
            if iteracje > 20:
                print('Byl problem ze zbieznoscia... zwracam aktualne rozwiazanie...')
                return (w)
        v = copy.deepcopy(w)
    print("Metoda gradientu")
    print("iteracje: " + str(iteracje))
    stop = time.time()
    print("Duration: " + str(stop - start))
    return (w)


print('Przyblizony punkt w ktorym osiagane jest minimum (metoda najszybszego spadku):\n',
      najszybszy_spadek([-5, 5], 1))


def metoda_Newtona(p):
    # p -- punkt poczatkowy,
    # to musi byc np.array kolumnowy, czyli np. np.array([[0], [0]])
    start = time.time()
    eps = 10 ** (-12)  # dokladnosc przyblizenia
    iteracje = 0  # liczba iteracji
    v = p  # pierwsze przyblizenie

    # funkcja do minimalizacji
    # f = (1-x)^2 + 100*(y-x^2)^2

    # wyliczamy gradient tej funkcji
    # grad(f) = [-2*(1-x) - 400*(y-x)*x, 200*(y-x^2)]

    # wyliczamy Hessian:
    # Hess(f) = [[2 + 800*x^2 -400(y-x^2), -400*x], [-400*x, 200]]

    # macierz odwrotna do macierzy Hessego
    # Hess(f)^(-1) = 1/((2+800*x)*200 - (400*x)^2)* [[200 , 400*x], [400*x, 2 + 800*x]]
    # 1 / ((2 + 800 * x) * 200 - (400 * x) ^ 2) * [[200, 400 * x], [400 * x, 2 + 800 * x]]
    # v - to wczesniejsze przyblizenie,

    # x = np.linspace(-5, 5, 35)
    # X, Y = np.meshgrid(x, x)
    # ax = plt.subplot(111, projection='3d')
    # ax.plot_surface(X, Y, rosen([X, Y]))
    # ax.set_xlabel("x", rotation=0, linespacing=3)
    # ax.set_ylabel("y", rotation=0, linespacing=3)
    # ax.set_zlabel('f(x)', rotation=0, linespacing=3)

    iteracje = 0
    d = [1, 1]  # wektor na start funkcji while

    while (math.sqrt(d[0] ** 2 + d[1] ** 2) > eps and iteracje < 100):
        # dopoki nie osiagniemy oczekiwanej dokladnosci
        # lub nie przekroczymy zadanej maksymalnej liczby iteracji
        iteracje += 1

        f = (1 - v[0, 0]) ** 2 + 100 * (v[1, 0] - v[0, 0] ** 2) ** 2
        print("iteracja = " + str(iteracje))
        print("x = " + str(v[0, 0]) + "   y = " + str(v[1, 0]) + "   f = " + str(f))

        # ax.plot(v[0, 0], v[1, 0], int(f), markerfacecolor='r', markeredgecolor='r', marker='o', markersize=15,
        #        alpha=0.6)

        gradient = np.array(
            [[-2 * (1 - v[0, 0]) - 400 * (v[1, 0] - v[0, 0] ** 2) * v[0, 0]], [200 * (v[1, 0] - v[0, 0] ** 2)]])

        odwrotna_macierz_Hess = -1.0 / (200 * (200 * (v[0, 0] ** 2) - 200 * v[1, 0] + 1)) * np.array(
            [[-100, -200 * v[0, 0]], [-200 * v[0, 0], -600 * (v[0, 0] ** 2) + 200 * v[1, 0] - 1]])

        d = - np.matmul(odwrotna_macierz_Hess, gradient)
        print("d = " + str(d))
        v = v + d  # glowne rownanie metody najmniejszego spadku

    print("Metoda Newtona")
    print("iteracje: " + str(iteracje))
    stop = time.time()
    print("Duration: " + str(stop - start))
    return (v)


print('\n')
print('Przyblizony punkt w ktorym osiagane jest minimum (metoda Newtona):\n',
      metoda_Newtona(np.array([[-0.3], [3.33]])))

# plt.show()
