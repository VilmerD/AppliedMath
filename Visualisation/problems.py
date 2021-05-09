import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton
import matplotlib.pyplot as plt


def problem1():
    def problem1u0():
        # The x-axis
        v0, u_max = 100, 100
        xx1 = np.linspace(-1, 0, 100, endpoint=True).reshape((100, 1))
        xx2 = np.linspace(0, 1, 100, endpoint=True).reshape((100, 1))
        xx = np.vstack((xx1, xx2))

        # The function
        u01 = u_max * np.ones((100, 1))
        u02 = np.zeros((100, 1))
        u0 = np.vstack((u01, u02))

        # Plotting
        fig, ax = plt.subplots()
        ax.plot(xx, u0)

        # Labels
        plt.xlabel('$x$ [km]')
        plt.ylabel('$u_0(x)$ [cars/km]')

        figureSaver(fig, 'u0_problem1.png')
        fig.show()

    def problem1epsilon():
        u0, u_max = 100, 100
        eps = 1e-2
        # The x-axis
        xx1 = np.linspace(-0.1, -eps, 100, endpoint=True).reshape((100, 1))
        xx2 = np.linspace(-eps, eps, 100, endpoint=True).reshape((100, 1))
        xx3 = np.linspace(eps, 0.1, 100, endpoint=True).reshape((100, 1))
        xx = np.vstack((xx1, xx2, xx3))

        # The function
        u01 = u_max * np.ones((100, 1))
        u02 = np.linspace(u_max, 0, 100).reshape((100, 1))
        u03 = np.zeros((100, 1))
        u0 = np.vstack((u01, u02, u03))

        # Plotting
        fig, ax = plt.subplots()
        ax.plot(xx, u0)

        # Labels
        plt.xlabel('$x$ [km]')
        plt.ylabel('$u_0(x)$ [cars/km]')
        plt.xticks((-eps, 0, eps), [r'$-\epsilon$', '0', r'$\epsilon$'])
        plt.yticks((0, u_max), ['0', '$u_{max}$'])

        figureSaver(fig, 'u0_problem1eps.png')
        fig.show()

    def problem1lines(plotFan):
        xMax, tMax = 1, 0.01
        nLines = 20
        xx = np.linspace(-xMax, xMax, nLines, endpoint=True).reshape((nLines, 1))
        tt = np.linspace(0, tMax, 100).reshape((100, 1))
        vz = np.ones((int(nLines / 2), 1))
        lines1 = -100 * vz @ tt.T + xx[0:int(nLines / 2)]
        lines2 = 100 * vz @ tt.T + xx[int(nLines / 2):]

        # Plotting
        fig, ax = plt.subplots()
        ax.plot(lines1.T, tt, 'r' if not plotFan else 'k')
        ax.plot(lines2.T, tt, 'g' if not plotFan else 'k')
        if plotFan:
            nLinesFan = 10
            aMin = np.arctan(100 / (xMax / tMax))
            aMax = np.pi - aMin
            angles = np.linspace(aMin, aMax, nLinesFan).reshape((nLinesFan, 1))
            vnz = (xMax * np.cos(angles)) / (tMax * np.sin(angles))
            fanLines = vnz @ tt.T
            ax.plot(fanLines.T, tt, 'k')
            name = 'u0_problem1LinesAndFan.png'
        else:
            ax.text(-0.5, 0.005, '$u = 100$', color='r')
            ax.text(0.3, 0.005, '$u = 0$', color='g')
            name = 'u0_problem1Lines.png'
        plt.xlabel('x [km]')
        plt.ylabel('t [h]')
        plt.xlim((-1, 1))

        figureSaver(fig, name)
        fig.show()
    problem1u0()
    problem1lines(False)
    problem1epsilon()
    problem1lines(True)

def problem2():
    v0, vi, u_max = 100, 70, 100
    L = 0.5
    ui = u_max * (1 - vi / v0)
    xx1 = np.linspace(-1, 0, 100).reshape((100, 1))
    xx2 = np.linspace(0, L, 100).reshape((100, 1))
    xx3 = np.linspace(L, 1, 100).reshape((100, 1))
    xx = np.vstack((xx1, xx2, xx3))

    u01 = ui * np.ones((100, 1))
    u02 = u_max * np.ones((100, 1))
    u03 = np.zeros((100, 1))
    u0 = np.vstack((u01, u02, u03))

    fig, ax = plt.subplots()
    ax.plot(xx, u0)
    plt.xlabel('$x$ [km]')
    plt.ylabel('$u_0(x)$ [cars/km]')
    figureSaver(fig, 'u0_problem2.png')
    fig.show()


def problem3():
    v0, vi, u_max = 100, 70, 100
    L = 0.5
    ui = u_max * (1 - vi / v0)
    xx1 = np.linspace(-0.5, 0, 100).reshape((100, 1))
    xx2 = np.linspace(0, L, 100).reshape((100, 1))
    xx3 = np.linspace(L, 1, 100).reshape((100, 1))
    xx = np.vstack((xx1, xx2, xx3))

    u01 = ui * np.ones((100, 1))
    u02 = u_max * np.ones((100, 1))
    u03 = np.zeros((100, 1))
    u0 = np.vstack((u01, u02, u03))

    fig, ax = plt.subplots()
    ax.plot(xx, u0)
    plt.xlabel('$x/L$  [1]')
    plt.ylabel('$u_0/u_{max}$  [1]')
    plt.xticks((0, 0.5), ['0', '1'])
    plt.yticks((0, 30, 100), ['0', r'$1 - \frac{v_i}{v_o}$', '$1$'])
    figureSaver(fig, 'u0_problem2.png')
    fig.show()


def figureSaver(fig, name, shape=(7, 2), bottom=0.25, left=0.15):
    fig.tight_layout()
    fig.subplots_adjust(bottom=bottom)
    fig.subplots_adjust(left=left)
    fig.set_size_inches(shape)
    fig.savefig(name, format='png', dpi=100)


problem1()