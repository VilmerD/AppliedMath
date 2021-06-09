from Visualisation.funcs import *

if __name__ == '__main__':
    # Problem parameters
    v0, vi, u_max = 1, 0.7, 1
    L = 0.1
    v = lambda u: v0 * (1 - u / u_max)
    f = lambda u: (v(u) * u)
    fp = lambda u: v0 * (1 - 2 * u / u_max)

    # Plotting parameters
    # tspan = 0, L/vi*1.1
    # xrange = -(v0 - vi)*L/vi*1.1, 0.11
    # plotCharacteristicLines(f, fp, v, v0, vi, u_max, L, tspan, xrange, Nx1=30, plot_cars=0, nFan=15)
    
    plotFluxFunction(f)
