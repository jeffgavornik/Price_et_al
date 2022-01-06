function [pvalue] = RunPermutationTest(data,N,func)
% Inputs:
%    data - vector of all data points to be compared in permutation test
%        (e.g. data from N Day 1 points vertically stacked on data from M
%        Day x points)
%    N - scalar, number of data points in first group
%    func - function handle to compute test statistic
%
% Outputs:
%    pvalue - two-sided permutation test p-value

if nargin<3
    func = @(x,N) mean(x(1:N))-mean(x(N+1:end));
end

dataStat = func(data,N);
NM = length(data);

nPerms = 1e6;
permStats = zeros(nPerms,1);
for kk=1:nPerms
    inds = randperm(NM,NM);
    permdata = data(inds);
    permStats(kk) = func(permdata,N);
end
pvalue = 1-mean(abs(permStats)<=abs(dataStat));


end
