function [U1_final, U2_final, P1_final, P2_final, V_final] = SMFH( X1, X2,param, d ,show)
%% random initialization
% show = 1;
X1 = X1';X2 = X2';
[row, col] = size(X1);
[~, colt] = size(X2);
P1 = abs(rand(d, col));
P2 = abs(rand(d, colt));
threshold = 1e-1;
lastF = 1e8;
iter = 1;
% ===================== parameter defined ========================
Lr = param.S;
lambda = param.lambda;
mu = param.mu;
alpha = param.alpha;
gamma = param.gamma;
NSamp = param.NSamp;
gamma1 = param.gamma1;
yeta = param.yeta;

% ===================== initializtion ========================
Norm = 2;
NormV = 1;
U1 = abs(rand(col,d));
U2 = abs(rand(colt,d));
V = abs(rand(row,d));
[U1,V] = NormalizeUV(U1, V, NormV, Norm);
[U2,V] = NormalizeUV(U2, V, NormV, Norm);
%% compute iteratively
while (true)
    % ===================== update V ========================
    XU1 = X1 *U1;  % mnk or pk (p<<mn)
    UU1 = U1'*U1;  % mk^2
    VUU1 = V*UU1; % nk^2
    XU2 = X2 *U2;  % mnk or pk (p<<mn)
    UU2 = U2'*U2;  % mk^2
    VUU2 = V*UU2; % nk^2
    
    % ===================== sampling ========================
    rp = randperm(size(Lr,1)); rp = rp(1:NSamp);
    ind = sparse(NSamp, size(Lr,1));
    for nnn = 1:NSamp
        ind(nnn,rp(nnn)) = 1;
    end
    L_tmp = Lr(rp,:); Vtmp = V(rp,:);
    NL = NSamp;
    XU = lambda * XU1 + (1-lambda) * XU2 + alpha*NL^-.1 * ind' * (L_tmp * (L_tmp' * Vtmp));
    VUU = lambda * VUU1 + (1-lambda) * VUU2 + alpha*NL^-.1 * ind' * bsxfun(@times, L_tmp * (sum(L_tmp,1))', Vtmp);

    PX = mu *  (X1 * P1' + X2 * P2');
    
    XU = XU +  PX + gamma1 * 4 * V;
    VUU = VUU + 2 * mu * V  + gamma1 * 4 * V * ( V' * V);

    V = V.*(XU./max(VUU,1e-10));
    % ===================== update U1 ========================
    XV = X1'*V;   % mnk or pk (p<<mn)
    VV = V'*V;  % nk^2
    UVV = U1*VV; % mk^2
    U1 = U1.*(XV./max(UVV,1e-10)); % 3mk
    % ===================== update U2 ========================
    XV = X2'*V;   % mnk or pk (p<<mn)
    VV = V'*V;  % nk^2
    UVV = U2*VV; % mk^2
    U2 = U2.*(XV./max(UVV,1e-10)); % 3mk
    [U1,V] = NormalizeUV(U1, V, NormV, Norm);
    [U2,V] = NormalizeUV(U2, V, NormV, Norm);
    % ===================== update P1 and P2 ========================

    S = L_tmp * L_tmp';
    DCol = full(sum(S,2));
    D = spdiags(DCol,0,NSamp,NSamp);
    L = D - S;
    D_mhalf = spdiags(DCol.^-.5,0,NSamp,NSamp) ;
    L = D_mhalf*L*D_mhalf;
    NL = NSamp;
    P1 = V' * X1 / (X1' * X1 + yeta*NL^-.1 * X1(rp,:)' * L * X1(rp,:) + gamma * eye(col));
    P2 = V' * X2 / (X2' * X2 + yeta*NL^-.1 * X2(rp,:)' * L * X2(rp,:) + gamma * eye(colt));
    
    % ===================== validation ========================
    L_tmp = Lr(rp,:); Vtmp = V(rp,:);
    [norm1, normNMF,  normGraph] = CalculateObjRand(X1, X2, U1, U2, V, Vtmp, L_tmp,lambda,alpha);
  
    norm2 = mu * norm(V - X1 * P1', 'fro');
    norm3 = mu * norm(V - X2 * P2', 'fro');
    norm4 = gamma * (norm(P1, 'fro') + norm(P2, 'fro'));
    currentF = norm1 + norm2 + norm3 + norm4;
    if ((lastF - currentF) < threshold ) || iter > 50 || normGraph < 100
        if iter > 5
            return;
        end
    end
    if show ~= 0
        fprintf('\nobj at iteration %d: %.4f %.4f %.4f\n', iter, normNMF, normGraph, norm1);
    end
    U1_final = U1;
    U2_final = U2;
    P1_final = P1;
    P2_final = P2;
    V_final = V;

    iter = iter + 1;
    lastF = currentF;
end
return;

function [U, V] = NormalizeUV(U, V, NormV, Norm)
K = size(U,2);
if Norm == 2
    if NormV
        norms = max(1e-15,sqrt(sum(V.^2,1)))';
        V = V*spdiags(norms.^-1,0,K,K);
        U = U*spdiags(norms,0,K,K);
    else
        norms = max(1e-15,sqrt(sum(U.^2,1)))';
        U = U*spdiags(norms.^-1,0,K,K);
        V = V*spdiags(norms,0,K,K);
    end
else
    if NormV
        norms = max(1e-15,sum(abs(V),1))';
        V = V*spdiags(norms.^-1,0,K,K);
        U = U*spdiags(norms,0,K,K);
    else
        norms = max(1e-15,sum(abs(U),1))';
        U = U*spdiags(norms.^-1,0,K,K);
        V = V*spdiags(norms,0,K,K);
    end
end

function [obj, obj_NMF, obj_Lap] = CalculateObjRand(X1, X2, U1, U2, V, V1, Lr,lambda,alpha)
MAXARRAY = 500*1024*1024/8; % 500M. You can modify this number based on your machine's computational power.
mn = numel(X1);

if mn < MAXARRAY
    dX1 = U1*V'-X1';
    dX2 = U2*V'-X2';
    obj_NMF = lambda*norm(dX1, 'fro')+(1-lambda)*norm(dX2, 'fro');
end

if V == 0
    obj_Lap = 0;
else
    D = V1' * bsxfun(@times, Lr * (sum(Lr,1))', V1);
    W = ((V1'* Lr) * (Lr' * V1));
    % obj_Lap = sum(sum(D-W));
    obj_Lap = norm(D-W, 'fro');
end
obj = obj_NMF+ alpha * obj_Lap;
