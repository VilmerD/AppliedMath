#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:28:59 2021

@author: vilmer
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton
import matplotlib.pyplot as plt


"""
Plot the characteristic lines given by the flows flow, 
from the positions x0
"""
def findIntersectAndPlot(flow, x0, t_range, s, spr, ax):
    N, = x0.shape
    M, = t_range.shape
    t0, tf = t_range[[0, -1]]
    t_guess = np.array([(tf - t0)/2])
    # Find intersection, if any, between lines and shock
    for i in np.arange(0, N):
        Li = lambda t: flow[i] * t + x0[i]
        Fi = lambda t: s(t[0]) - Li(t[0])
        Fpi = lambda t: spr(t[0]) - flow[i]
        try:
            ts = newton(Fi, t_guess, fprime=Fpi)
            ts = min(ts, tf)
            ts = ts if ts >= 0 else tf
        except RuntimeError:
            ts = tf
        tti = np.linspace(t0, ts, M)
        ax.plot(flow[i] * tti + x0[i], tti, 'k')


"""
Plots the characteristic lines
"""
def plotCharacteristicLines(f, fp, v, v0, vi, u_max, L, trange, xrange, Nx1 = 40, plot_cars=0, nFan=15):
    # Setup
    t0, tf = trange
    x_min, x_max = xrange

    # Setup initial condition
    u_minus = u_max * (1 - vi / v0)

    def u0(x):
        return u_minus * (x < 0) + u_max * (x < L) * (x >= 0) + 0 * (x >= L)

    # Setup t-axis
    Nt = 100
    tt = np.linspace(t0, tf, num=Nt)

    # Setup plot and axi
    fig, ax = plt.subplots()

    # Make xaxis
    # Distance between two lines on x-axis
    xx_min, xx_max = x_min*10, x_max
    dx1 = (xx_max - xx_min) / Nx1

    # Shortest distance between two lines should be same for
    # all lines, defined as d
    d = abs(dx1 * np.sin(np.pi + np.arctan(1 / fp(u_minus))))
    xx1 = np.linspace(xx_min, 0, num=Nx1)

    # Ensure distance between two lines is exactly d
    dx2 = abs(d / np.sin(np.arctan(1 / fp(u_max))))/3
    Nx2 = int((xx_max - 0) / dx2)
    xx2 = np.linspace(0, xx_max, num=Nx2)

    # Stack the points
    xx = np.hstack((xx1, xx2))

    # Make characteristic lines
    u0_val = u0(xx)
    flowu0 = fp(u0_val)
    x0u0 = xx.copy()

    # Make fan
    theta0 = np.arctan(1/(v0/(x_max/tf)))
    theta_n = np.pi - theta0
    angles = np.linspace(theta0, theta_n, num=nFan + 1)[1:]
    flow_fan = (x_max * np.cos(angles)) / (tf * np.sin(angles))
    x0fan = L * np.ones((nFan,))

    # Stitch together fan and lines from u0
    flow = np.hstack((flowu0, flow_fan))
    x0 = np.hstack((x0u0, x0fan))

    tc = L/vi
    xc = -(v0 - vi)*tc

    def ufan(t, x):
        if t > 0:
            return (u_max / 2) * (1 - (x - L) / (t * v0))
        else:
            return (1/2)*(x == 0)

    # Shock wave
    def s(t):
        return -(v0 - vi)*t*(t <= tc) + ((2*vi - v0)*t - 2*np.sqrt(abs(vi*L*t)) + L)*(t > tc)

    def spr(t):
        return -(v0 - vi)*(t <= tc) + 1/2*((2*vi - v0) + (s(t) - L)/t)*(t > tc)

    def u_sol(t, x):
        # Lines
        l1 = (x < s(t))
        l2 = (x < L - v0 * t) * (t < tc) + (x < s(t)) * (t >= tc)
        l3 = (x < L + v0 * t)

        # Solution
        u1 = u_minus            *  l1 *  l2 *  l3
        u2 = u_max              * ~l1 *  l2 *  l3
        u3 = ufan(t, x)         * ~l1 * ~l2 *  l3
        u4 = 0                  * ~l1 * ~l2 * ~l3
        return u1 + u2 + u3 + u4

    ttshock = np.linspace(0, tf, Nt)
    findIntersectAndPlot(flow, x0, ttshock, s, spr, ax)

    # Finish plotting
    ax.plot(s(ttshock).reshape((Nt,)), ttshock, 'b--')

    if plot_cars > 0:
        x_car = np.linspace(x_min, L, num=plot_cars)
        xp = lambda t, x: v(u_sol(t, x))
        for x0 in x_car:
            sol = solve_ivp(xp, trange, [x0], method='RK45', dense_output=True, max_step=(tf - t0)/Nt)
            plt.plot(sol.sol(tt).reshape((Nt,)), tt, 'r')

    # Plotting dot at tc
    if tf < 8*tc: plt.plot(xc, tc, 'ro')

    # Tick-marks
    if tf < 8*tc:
        xticks = [0, L, xc]
        xtickmarks = ['0', 'L', '$x_c$']
        yticks = [0, tc, tf]
        ytickmarks = ['0', '$t_c$', '$t_f$']
        name = 'problem2Lines1.png'
    else:
        xticks = [0, L]
        xtickmarks = ['0', 'L']
        yticks = [0, tf]
        ytickmarks = ['0', '$t_f$']
        name = 'problem2Lines2.png'
    plt.xticks(xticks, xtickmarks)
    plt.yticks(yticks, ytickmarks)

    # Fixing axis and labels
    plt.xlim(xrange)
    plt.ylim(trange)
    plt.xlabel('x')
    plt.ylabel('t')

    # Preparations for saving figure
    fig.tight_layout()
    shape, bottom, left = (7, 2), 0.25, 0.10
    fig.subplots_adjust(bottom=bottom)
    fig.subplots_adjust(left=left)
    fig.set_size_inches(shape)
    fig.savefig(name, format='png', dpi=100)
    plt.show()


"""
Plots the paths of some cars given the max velocity v0 and the initial velocity vi
"""
def plotPaths(v0, vi):
    fig, ax = plt.subplots()
    t0, tf, Nt = 0, 2, 100
    tt = np.linspace(t0, tf, num=Nt)
    t_vec = tt.reshape((Nt, 1))

    x_min, x_max, Nx = -vi * tf / 4, v0 * tf / 4, 20
    xx = np.linspace(x_min, 0, num=Nx)

    for c in np.arange(0, Nx):
        xxc = xx[c]
        tsc = -xxc / v0
        p_stop = xxc * (t_vec < tsc)
        p_move = (v0 * t_vec - 2 * np.sqrt(-v0 * xxc * t_vec)) * (t_vec >= tsc)
        ax.plot(p_stop + p_move, t_vec, 'k')
    plt.xlim((x_min, x_max))
    plt.ylim((t0, tf))
    plt.xlabel('cars position x [1]')
    plt.ylabel('time [1]')
    plt.show()


def plotFluxFunction(f):
    fig, ax = plt.subplots()
    uu = np.linspace(0, 1, 100)
    ax.plot(uu, f(uu))

    # Shock speed
    um, up = 0.3, 1
    km = 0
    for k in range(0, km):
        upk = up + k*(um - up)/km
        l = (f(upk) - f(um))/(upk - um)*(uu - upk) + f(upk)
        ax.plot(uu, l, 'k')
        ax.plot(upk, f(upk), 'ro', um, f(um), 'ro')

    # Adding labels
    plt.xlabel('$u$')
    plt.ylabel('$f(u)$')

    # Adjusting tick marks
    xticks, xtickmarks = [0, 1], ['0', r'$u_{max}$']
    yticks, ytickmarks = [0, 1/4], ['0', r'$\frac{1}{4}v_0u_{max}$']
    plt.xticks(xticks, xtickmarks)
    plt.yticks(yticks, ytickmarks)

    # Making plot ready to save
    fig.tight_layout()
    shape, bottom, left = (5, 3), 0.15, 0.20
    fig.subplots_adjust(bottom=bottom)
    fig.subplots_adjust(left=left)
    fig.set_size_inches(shape)
    fig.savefig('fluxFunction.svg', format='svg', dpi=180, transparent=True)
