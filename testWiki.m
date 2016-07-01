clear all; warning off; clc;
load myWiki;
bit = 64; % the number of hash bits
%% precessing
if size(L_tr,2) == 1
    S = zeros(size(L_tr,1),max(L_tr));
    for i = 1:max(L_tr)
        ind = find(L_tr==i);
        S(ind,i) = 1;
    end
else
    L_tr(L_tr <= 0) = 0;
    L_tr(L_tr > 0) = 1;
    % %normalize the label matrix
    Length = sqrt(sum(L_tr.^2, 2));
    Length(Length == 0) = 1e-8; % avoid division by zero problem for unlabeled rows
    Lambda = 1 ./ Length;
    S = diag(sparse(Lambda)) * L_tr;
end
I_tr = normalize1(I_tr);T_tr = normalize1(T_tr);
I_te = normalize1(I_te);T_te = normalize1(T_te);
%% parameters setting
param.S = (S);
param.lambda = 0.5; param.mu = 40; param.alpha = 2;
param.gamma = 1e-4; param.gamma1 = 1e-4;
param.NSamp = 1000; param.yeta = 0.001;
%% coding
[~, ~, Wx, Wy, Y] = SMFH(I_tr', T_tr', param, bit ,1);
B1 = sign((bsxfun(@minus,Y , mean(Y,1))));
tB1 = sign((bsxfun(@minus,I_te * Wx', mean(Y,1))));
B2 = B1;
tB2 = sign((bsxfun(@minus,T_te * Wy' , mean(Y,1))));
%% evaluation
sim_it = B1 * tB2'; sim_ti = B2 * tB1';
map_t2i = mAP_wiki(sim_it,L_tr,L_te,0)
map_i2t = mAP_wiki(sim_ti,L_tr,L_te,0)

