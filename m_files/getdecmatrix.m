%% 读取label和output文件
clear
clc
name = 'CSZUL00000001_01'; %name of image to test.
numlabels = 100; %检查labels时候的训练图像序号
haha = double(imread(['../0112pic/',name,'.png']));

path1 = '../0112rs/'; %prediction输出路径
fp = fopen([path1,name,'.txt'],'r');
output = fscanf(fp,'%f',[1,inf]); 
fclose(fp);
for i = 1:size(output,2)  %调整越界回归输出
    if output(i) > 179
        output(i) = 179;
    elseif output(i)<0
        output(i) = 0;
    end
end
myresult = round(output/179*254);

%% 生成方向场图
% label = trueresult; %true
label0 = myresult; %prediction
rr = 1;
cell0 = cell(1,rr);
cell00 = cell(1,rr);
for l = 1:rr
%     cell0{l} = reshape(label(l,:),labelsize,labelsize)'; %cell0 存储label 1：400
    cell00{l} = reshape(label0(l,:),64,64)'; %cell0 存储my 1：400
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

%         result0(n0:n0+15,m0:m0+15) = cell0{kk}(k0,q0)*ones(16)*pi/254;  %label
        result00(n0:n0+15,m0:m0+15) = cell00{kk}(k0,q0)*ones(16)*pi/254; %my
        q0=q0+1;
    end
    k0 = k0+1;
    q0 = 1;
end
%result0 = result0*pi/180;
%mask0 = result0>0.1;
%maskind0 = find(mask0);
%haha = haha - mean(haha(maskind0));
%normim0 = haha/std(haha(maskind0));  
%normim0 = zeros(160,160);
 % Determine ridge orientations
   % [orientim0, reliability0] = ridgeorient(normim0, 1, 5, 5);
    %figure;
    %figure;
    %if(kk == 110)
%     plotridgeorient(result0, 8,haha, 3, 'r'); %画出标签在原图的下的效果图
    plotridgeorient(result00, 8,haha, 2, 'b');  %画出方向场在原图上的可视化效果图

    %end
    %show(reliability,6)
end