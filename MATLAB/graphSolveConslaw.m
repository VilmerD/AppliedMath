function L = graphSolveConslaw(fp, u0, xx, tt)
% lines = graphSolveConslaw(f, u0, x0) graphically solves the conservation
% law for a given flux function and initial conditions
n = length(u0);
k = fp(u0)';



t0 = tt(1); tend = tt(end);
nt = length(tt);

tc = tend*ones(n, 1);
T = zeros(n, nt);

for i = 1:n
    ki = k(i);
    mi = xx(i);
    for j = i:n
        kj = k(j);
        mj = xx(j);
        if abs(ki - kj) > 1e-6
            tcij = (mj - mi)/(ki - kj);
            tc(j) = max(0, min(tc(j), tcij));
        end
    end
end

L = k * tt + xx'.*(tt < tc);
end