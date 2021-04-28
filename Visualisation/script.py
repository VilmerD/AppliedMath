from AppliedMath.Visualisation.funcs import *

if __name__ == '__main__':
    v0, vi, u_max = 1, 0.7, 1
    L = 3
    v = lambda u: v0 * (1 - u / u_max)
    f = lambda u: v(u) * u
    fp = lambda u: v0 * (1 - 2 * u / u_max)
    u_sol = plotCharacteristicLines(f, fp, v0, vi, u_max, L)