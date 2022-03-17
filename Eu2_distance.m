function D = Eu2_distance(a,b)
% 计算样本点之间距离的平方，函数返回距离平方矩阵，n*m
% a,b ： 数据矩阵，每一列为一个样本
% D   ： 样本间的距离矩阵 n*m
% n   ： 矩阵 a 的样本数
% m   ： 矩阵 b 的样本数
% d   ： 样本的维度

[d1,n] = size(a);
[d2,m] = size(b);

if d1~=d2
    error('数据维度不同，无法计算')
end
D = zeros(n,m);
for i = 1:m
    D(:,i) = sum(((a-b(:,i)).^2),1);
end
