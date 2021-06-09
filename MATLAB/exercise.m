%% Exercise 9
xmin = -4; xmax = 4;
xx = linspace(xmin, xmax); 
u0 = @(x) 100*heaviside(x);

figure(1)
anisol(u0(xx), xx, @u9, 10/60)
%% Project
fp = @(u) 100 - 2*u;

%% Task 3
u0 = @(x) 30*heaviside(-x) + 100*heaviside(x).*heaviside(-x+0.1);

figure(1)
xx = linspace(-1, 0.2, 300);
plot(xx, u0(xx));

figure(2)
tend = 2/60;
graphSolveConslaw(fp, u0, [-1, 0.2], tend)
hold on;
tt = linspace(0, tend);
plot(-30*tt, tt, 'r')

%% Testing solutions
figure(3)
xmin = -.1; xmax = .2;
xx = linspace(xmin, xmax, 3000); 
anisol(u0(xx), xx, @u3, 1/680)

%% Solution to ex9
function u = u9(t, x)
h1 = heaviside(-100*t - x);
h2 = heaviside(x - 100*t);

u = h1*100 + (100 - x./t)/2.*~h1.*~h2;
end

function u = u3(t, x)
h1 = heaviside(-30*t-x);
h2 = heaviside(0.1-100*t-x);
h3 = heaviside(0.1+100*t-x);

u = h1*30 + ~h1.*h2.*100 + ((x - 0.1)/2./t + 50).*~h1.*~h2.*h3;
end

