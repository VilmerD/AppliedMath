%% Seting up dif
v0 = 100;
umax = 100;
f = @(u) v0*u*(1 - u/umax);
fp = @(u) v0*(1 - 2*u/umax);
u0 = @(x) 30*heaviside(-x) + 100*heaviside(x).*heaviside(-x+0.1);

%% Plot
a = [-60/700 0.1 0 tend];
plot(xshock, t);
axis(a);
xlabel('x [km]')
ylabel('t [h]')

%% Shock speed
xpeval = xp(xshock, t);
xpevaltol = 1e2;
id = find(abs(xpeval) < xpevaltol);
figure(2);
plot(tt(id), xpeval(id));
xlabel('$\dot{x}(t)$', 'Interpreter', 'Latex')
ylabel('t [h]')

%% Animation
tend = 1/10; nt = 5e3;
tt = linspace(t0, tend, nt);
[t, x, us] = su3(tend, nt);
figure(3)
xmin = -.2; xmax = 100*tend; nx = 1e3;
xx = linspace(xmin, xmax, nx); 
anisol(u0(xx), xx, us, tt)

%%
function [tshock, xshock, u3] = su3(tend, nt)
    xp = @(t, x) (1600*t.^2 - (0.1-x).^2)./(0.2*t-2*x.*t+80*t.^2);

    % Initial values
    t0 = 1/700;
    x0 = -30/700;

    tt = linspace(t0, tend, nt);
    [tshock, xshock] = ode15s(xp, tt, x0);
    
    function xi = xs(t)
        xi = xshock(abs(tshock - t) < (tend - t0)/(2*nt));
    end
    
    function u = u3s(t, x)
        xi = xs(t);
        
        x1 = @(t)           -30*t*(t < t0) + xi*(t >= t0);
        x2 = @(t)   (0.1 - 100*t)*(t < t0) + xi*(t >= t0);
        x3 = @(t) 0.1 + 100*t;
        h1 = heaviside(x - x1(t));
        h2 = heaviside(x - x2(t));
        h3 = heaviside(x - x3(t));

        u = ~h1*30 + h1.*~h2*100 + h2.*~h3.*(0.1 - x + 100*t)./(2*t);
    end
    
    u3 = @u3s;
end