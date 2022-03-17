function [result,Sort_anchor,repp] = PSSCNCD(X,q,k0,label,alpha_u,alpha_l)

% label: ������֤�������ܵ���ʵ��ǩ
% X: ԭʼ���ݾ���n*d
% q = ê����� k0 = ����ͼ������ c = �����
result = zeros(1,5);   % ��¼������ָ���Լ�bkhk����ʱ�����ʱ��
% alpha_lĬ��Ϊ0��alpha_uĬ��ΪС��1����
% alpha_u: bigger the value, lower the ability to discover novel class ��ýӽ�1
c = length(unique(label));

if nargin < 6
    alpha_l = 0;
end
if nargin < 5
    alpha_u = 0.99;
end


%% BKHK���� O(ndlog(m)+nmd)
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
end                                                                   % �õ�m��ê���ʱ�临�Ӷ���ndlog(m)
for i = 1:m
    for j = 1:n
        if U_final(i,:) == X(j,:)
            Sort_anchor(i) = j;                                       % m��ê���Ӧ����ţ����㸴�Ӷ���nmd
        else
        end
    end
end
t_bkhk = toc;




%% ����ͼ����  O(nmd)
E_distance_m = Eu2_distance(X',U_final');   
[~,idx] = sort(E_distance_m,2);                              %��������ÿһ�У�ָʾ����ļ������� 
gamma = zeros(n,1);                                          %����gamma��������ͬ�������в�ͬ��gammaֵ
for i = 1:n
     if ismember(i,Sort_anchor)==0                           %���������㲻����anchor����ѡ��k0+1���ĵ�
        id = idx(i,1:k0+1);
        di = E_distance_m(i,id);
        gamma(i) = (k0*di(k0+1)-sum(di(1:k0))+eps)/2;        %gamma�ļ��㹫ʽ
     else
        id = idx(i,2:k0+2);                                  %��������������anchor����ѡ��k0+2���ĵ�
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
     


%% ������ල������ʼ��
r = randperm(m,1);                                           % ��m��ê�������ѡȡһ��ê��
eu_dist = zeros(n,1);
for i = 1:n
    eu_dist(i) = norm(X(i,:)-X(Sort_anchor(r),:));           % ��ê��������������ľ���  ʱ�临�Ӷ�nd
end
eu_dist(Sort_anchor(r)) = [];
[~,idm] = min(eu_dist);                                      % ѡȡ�����ê�������һ����
if idm >= Sort_anchor(r)
    idm = idm + 1;                                           % ѡ����һ����������ԭ����������
else
end



%% construct label matrix Y and similarity matrix W
Y_init = zeros(n,2);                                         % �����ʼ��ǩ����
Y_init(idm,1) = 1;                                           % ��ѡ���ĵ�һ����������ڵ�һ��
H = linspace(1,n,n);
H(idm) = [];
for i = 1:n-1
    Y_init(H(i),2) = 1; % ����������Ϊ����
end






%% Regularization parameter alpha/matrix U
alpha = zeros(n,1);
alpha(idm) = alpha_l;                                        % \alpha_l���ѱ�������й�
alpha(H) = alpha_u;                                          % \alpha_u���Ƿ��������й�
alpha = diag(alpha);
beta = diag(ones(n,1)-diag(alpha));



%% Conducting initial semisupervised framework
P = delta - Bip'*alpha*Bip; % m^2*n �� mn
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
        Y(H(hh),i+2) = 1;                 % δ�������ȫ��Ϊ����
    end
    for j = 1:i+1
        Y(repp(j),j) = 1;                 % ������������趨α��ǩ
    end
    
    
    
    
    
    %% update the matrix U and operate semisupervised framework
    if i < c-1
        alpha = zeros(n,1);
        alpha(repp(1:i+1)) = alpha_l;
        alpha(H) = alpha_u;
        alpha = diag(alpha);
        beta = diag(ones(n,1)-diag(alpha));
        P_main = delta - Bip'*alpha*Bip;     % m^2*n �� mn
        P_1_main = (alpha*Bip)/(P_main);     % mn m^3 m^2*n
        F = P_1_main*(Bip'*(beta*Y))+beta*Y; % 2mn 2mn 2n 2n
    else
        alpha_u = 1;
        alpha = zeros(n,1);
        alpha(repp) = alpha_l;
        alpha(H) = alpha_u;
        alpha = diag(alpha);
        beta_c = diag(ones(n,1)-diag(alpha));
        P_1_sep = Bip'*alpha*Bip;            % m^2*n �� mn
        P_sep = delta - P_1_sep;             % mn m^3 m^2*n
        P_2_sep = (alpha*Bip)/(P_sep);       % 2mn 2mn 2n 2n
        F = P_2_sep*(Bip'*(beta_c*Y))+beta_c*Y; % ʱ�临�Ӷ�Ϊmn
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
