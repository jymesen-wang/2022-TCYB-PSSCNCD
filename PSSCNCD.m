function [result,Sort_anchor,repp] = PSSCNCD(X,q,k0,label,alpha_u,alpha_l)

% label: 用于验证聚类性能的真实标签
% X: 原始数据矩阵n*d
% q = 锚点层数 k0 = 二部图近邻数 c = 类别数
result = zeros(1,5);   % 记录聚类三指标以及bkhk运行时间和总时间
% alpha_l默认为0，alpha_u默认为小于1的数
% alpha_u: bigger the value, lower the ability to discover novel class 最好接近1
c = length(unique(label));

if nargin < 6
    alpha_l = 0;
end
if nargin < 5
    alpha_u = 0.99;
end


%% BKHK方法 O(ndlog(m)+nmd)
m = 2^(q-1);
[n,d] = size(X);
BKHK = cell(m,q);
BKHK{1,1} = X;
tic
pause(1)
for i = 1:(q-1)
    for j = 1:2^(i-1)
        [BKHK{2*j-1,i+1},BKHK{2*j,i+1},~] = BKHK_onestep(BKHK{j,i});
    end
end
U_final = zeros(m,d);Sort_anchor = zeros(m,1);                        % the last layer of BKHK and Rank
for i = 1:2^(q-2)
    [BKHK{2*i-1,q},BKHK{2*i,q},U] = BKHK_onestep(BKHK{i,q-1}); 
    U_final(2*i-1,:) = U(1,:);
    U_final(2*i,:) = U(2,:);
end                                                                   % 得到m个锚点的时间复杂度是ndlog(m)
for i = 1:m
    for j = 1:n
        if U_final(i,:) == X(j,:)
            Sort_anchor(i) = j;                                       % m个锚点对应的序号，计算复杂度是nmd
        else
        end
    end
end
t_bkhk = toc;




%% 二部图构建  O(nmd)
E_distance_m = Eu2_distance(X',U_final');   
[~,idx] = sort(E_distance_m,2);                              %升序排列每一行，指示最近的几个距离 
gamma = zeros(n,1);                                          %构造gamma向量，不同的样本有不同的gamma值
for i = 1:n
     if ismember(i,Sort_anchor)==0                           %若该样本点不属于anchor集，选最k0+1近的点
        id = idx(i,1:k0+1);
        di = E_distance_m(i,id);
        gamma(i) = (k0*di(k0+1)-sum(di(1:k0))+eps)/2;        %gamma的计算公式
     else
        id = idx(i,2:k0+2);                                  %若该样本点属于anchor集，选最k0+2近的点
        di = E_distance_m(i,id);
        gamma(i) = (k0*di(k0+1)-sum(di(1:k0))+eps)/2;
     end
end
Bip = zeros(n,m);
for i = 1:n
    [Bip(i,:),~] = EProjSimplex_new(-E_distance_m(i,:)/(2*gamma(i))); % the sum of each row is 1,solve the problem (6) in FSC
end
delta = diag(sum(Bip));
%W = Bip/(delta)*Bip';
     


%% 搜索半监督方法初始化
r = randperm(m,1);                                           % 从m个锚点中随机选取一个锚点
eu_dist = zeros(n,1);
for i = 1:n
    eu_dist(i) = norm(X(i,:)-X(Sort_anchor(r),:));           % 该锚点与所有样本点的距离  时间复杂度nd
end
eu_dist(Sort_anchor(r)) = [];
[~,idm] = min(eu_dist);                                      % 选取距离该锚点最近的一个点
if idm >= Sort_anchor(r)
    idm = idm + 1;                                           % 选定第一个代表点对于原样本点的序号
else
end



%% construct label matrix Y and similarity matrix W
Y_init = zeros(n,2);                                         % 构造初始标签矩阵
Y_init(idm,1) = 1;                                           % 被选定的第一个代表点属于第一类
H = linspace(1,n,n);
H(idm) = [];
for i = 1:n-1
    Y_init(H(i),2) = 1; % 其余样本都为新类
end






%% Regularization parameter alpha/matrix U
alpha = zeros(n,1);
alpha(idm) = alpha_l;                                        % \alpha_l与已标记样本有关
alpha(H) = alpha_u;                                          % \alpha_u与是否发现新类有关
alpha = diag(alpha);
beta = diag(ones(n,1)-diag(alpha));



%% Conducting initial semisupervised framework
P = delta - Bip'*alpha*Bip; % m^2*n 和 mn
P_1 = (alpha*Bip)/(P);      % mn m^3 m^2*n
F = P_1*(Bip'*(beta*Y_init))+beta*Y_init; % 2mn 2mn 2n 2n



%% select the most representative point x_j
repp = zeros(c,1);
repp(1) = idm;
for i = 1:c-1
    i
    [~,idm_next] = max(F(:,i+1));
    repp(i+1) = idm_next;
    Y = zeros(n,i+2);
    H = linspace(1,n,n);
    H(repp(1:i+1)) = [];
    for hh = 1:n-i-1
        Y(H(hh),i+2) = 1;                 % 未标记样本全部为新类
    end
    for j = 1:i+1
        Y(repp(j),j) = 1;                 % 标记样本依次设定伪标签
    end
    
    
    
    
    
    %% update the matrix U and operate semisupervised framework
    if i < c-1
        alpha = zeros(n,1);
        alpha(repp(1:i+1)) = alpha_l;
        alpha(H) = alpha_u;
        alpha = diag(alpha);
        beta = diag(ones(n,1)-diag(alpha));
        P_main = delta - Bip'*alpha*Bip;     % m^2*n 和 mn
        P_1_main = (alpha*Bip)/(P_main);     % mn m^3 m^2*n
        F = P_1_main*(Bip'*(beta*Y))+beta*Y; % 2mn 2mn 2n 2n
    else
        alpha_u = 1;
        alpha = zeros(n,1);
        alpha(repp) = alpha_l;
        alpha(H) = alpha_u;
        alpha = diag(alpha);
        beta_c = diag(ones(n,1)-diag(alpha));
        P_1_sep = Bip'*alpha*Bip;            % m^2*n 和 mn
        P_sep = delta - P_1_sep;             % mn m^3 m^2*n
        P_2_sep = (alpha*Bip)/(P_sep);       % 2mn 2mn 2n 2n
        F = P_2_sep*(Bip'*(beta_c*Y))+beta_c*Y; % 时间复杂度为mn
    end
end
%% obtain clustering result
F = F(:,1:c);
[~,index] = sort(transpose(F),'descend');
labelnew = index(1,:);
labelnew = labelnew';
t_full = toc;
result(1:3) = ClusteringMeasure(label,labelnew);
result(4) = t_bkhk;result(5) = t_full;
