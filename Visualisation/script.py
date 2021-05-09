from Visualisation.funcs import *

if __name__ == '__main__':
    # Problem paramters
    v0, vi, u_max = 1, 0.7, 1
    L = 1
    v = lambda u: v0 * (1 - u / u_max)
    f = lambda u: v(u) * u
    fp = lambda u: v0 * (1 - 2 * u / u_max)
<<<<<<< HEAD

    # Plotting parameters
    tspan = 0, 30
    xrange = -3, 3
    plotCharacteristicLines(f, fp, v, v0, vi, u_max, L, tspan, xrange, Nx1=40, plot_cars=8)
=======
    # plotCharacteristicLines(f, fp, v0, vi, u_max, L)
    
    plotFluxFunction(f)
    
>>>>>>> 7004f648ad29b2f647abf5244e30ddaebda6efe3
