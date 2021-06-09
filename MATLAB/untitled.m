%%
xmax = 1;           tmax = 1/100;
aMin = atan(1/100); aMax = pi - aMin;
n = 12;

x = @(theta) xmax*cos(theta);
t = @(theta) tmax*sin(theta);
f = @(theta) sqrt(x(theta).^2 + t(theta).^2);
li = @(thetan, thetap) integral(f, thetap, thetan);

ltot = integral(f, aMin, aMax);
thetas(n) = pi/2;
thetap = 0;
for k = 1:n-1
    thetan = fzero(@(theta) li(theta, thetap) - ltot/n, k*(aMax - aMin)/n);
    thetas(k) = thetan;
    li(thetan, thetap)
    thetap = thetan;
end
tt = linspace(0, tmax);
vx = 1./tan(thetas);
lines = vx'*tt;
plot(lines', (ones(n, 1)*tt)', 'k')
axis([-xmax, xmax, 0, tmax])

%%
aMin = atan(100/(xmax/tmax)); aMax = pi - aMin;
angles = linspace(aMin, aMax, n);
vx = x(angles)./t(angles);
lines = vx'*tt;
plot(lines', (ones(n, 1)*tt)', 'k')
axis([-xmax, xmax, 0, tmax])