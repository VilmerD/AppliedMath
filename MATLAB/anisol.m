function anisol(u0, xx, u, tt)
nt = length(tt);
xmin = min(xx); xmax = max(xx);
l = plot(xx, u0);
for k = 2:nt
    uk = u(tt(k), xx);
    try
        clf(l, 'reset')
    catch 
        return
    end
    l = plot(xx, uk);
    axis([xmin, xmax, 0, 100])
    pause(0.0001)
end

