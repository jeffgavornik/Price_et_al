function [loglikelihood,CnInv] = MbTDRMarginal_Likelihood(S,B,D,N,X,Z,P,R,trials)
% marginal likelihood for MbTDR model
if R(1)>0
    newX = X{1};
    
    for pp=2:P
        if R(pp)>0
            newX = [newX,X{pp}];
        end
    end

else
    newX = X{2};
    for pp=3:P
        if R(pp)>0
            newX = [newX,X{pp}];
        end
    end
end
X = newX;

S = blkdiag(S{:});

U = X*S;

% Im = eye(M);
trials = logical(trials);
loglikelihood = 0;
CnInv = cell(N,1);
for nn=1:N
    UtU = U(trials(nn,:),:)'*U(trials(nn,:),:);
    [V,Q] = eig(UtU);
    Ir = ones(size(Q,1),1);
    diagQQ = diag(Q);
    
    CnInv{nn} = bsxfun(@times,V,1./(Ir+diagQQ./D(nn))')*V';
    
%     DnInv = Im./D(nn)-(U*CnInv*U')./(D(nn)*D(nn));
%     Cn{nn} = diag(Ir)+UtU./D(nn);
    logDnDet = 2*sum(log(diag(chol(diag(Ir)+UtU./D(nn)))))+sum(trials(nn,:))*log(D(nn));
    Q = Z(nn,trials(nn,:))'-B(nn);
    tmp = chol(CnInv{nn})*U(trials(nn,:),:)'*Q;
    loglikelihood = loglikelihood-0.5*logDnDet-...
        0.5*(Q'*Q)./D(nn)+0.5*(tmp'*tmp)./(D(nn)*D(nn));
end
end
