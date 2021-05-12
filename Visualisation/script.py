from AppliedMath.Visualisation.funcs import *

if __name__ == '__main__':
    # Problem parameters
    v0, vi, u_max = 100, 70, 100
    L = 0.1
    v = lambda u: v0 * (1 - u / u_max)
    f = lambda u: (v(u) * u)
    fp = lambda u: v0 * (1 - 2 * u / u_max)

    # Plotting parameters
    tspan = 0, 25*L/vi
    xrange = -0.2, 0.5
    plotCharacteristicLines(f, fp, v, v0, vi, u_max, L, tspan, xrange, Nx1=30, plot_cars=0, nFan=15)
    
    # plotFluxFunction(f)