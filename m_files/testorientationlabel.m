%% 读取label和output文件
clear
clc
labelsize = 20; %label矩阵的大小
num = 0; %读取的指纹和对应标签文件名

path = 'E:/fingerprint/1030/pic/'; %pics路径

% path1 = 'E:/fingerprint/1030/output'; %prediction输出路径
% path2 =['E:/fingerprint/1030/labels/',int2str(num)]; %labels路径
% path = 'E:/fingerprint/1216/pic/'; %pics路径

path1 = 'E:\fingerprint\1030\labels\'; %prediction输出路径

fp = fopen([path1,int2str(num),'.txt'],'r');
output = fscanf(fp,'%f',[1,inf]); 
fclose(fp);
myresult = round(output/127*254);
haha = double((imread([path,int2str(num),'.bmp'])));

%% 生成方向场图
label0 = myresult; %prediction
rr = 1;
cell00 = cell(1,rr);
for l = 1:rr
    cell00{l} = reshape(label0(l,:),20,20)'; %cell0 存储my 1：400
end
[rows0,cols0] = size(haha); %160 160
result0 = zeros(rows0,cols0);
result00 = zeros(rows0,cols0);

k0 = 1;
q0 = 1;
kk=1;
for kk = 1:rr
    n0=1;
    m0=1;
    k0=1;
for n0 = 1:8:rows0
    for m0 = 1:8:cols0  
        result00(n0:n0+15,m0:m0+15) = cell00{kk}(k0,q0)*ones(16)*pi/254; %my
        q0=q0+1;
    end
    k0 = k0+1;
    q0 = 1;
end
    plotridgeorient(result00, 8,haha, 2, 'b');  %画出方向场在原图上的可视化效果图

    %end
    %show(reliability,6)
end