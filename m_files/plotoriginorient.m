clear
clc
num = 3713;
path = 'E:/fingerprint/1030/pic/'; %pics
path1 = 'E:/fingerprint/1030/output'; %prediction
path2 =['E:/fingerprint/1030/labels/',int2str(num)]; %labels

%predictions
fp = fopen([path1,'.txt'],'r');
output = fscanf(fp,'%f',[1,inf]); %800维行向量
fclose(fp);
for i = 1:size(output,2)
    if output(i) > pi
        output(i) = pi;
    elseif output(i)<0
        output(i) = 0;
    end
end

%labels
fp = fopen([path2,'.txt'],'r');
trueresult = round(fscanf(fp,'%f',[1,inf])/127*254); %800维行向量

myresult = round(output/pi*254);
haha = double(imread([path,int2str(num),'.bmp']));
% haha = double(imread(['E:/fingerprint/0921data/test/',int2str(1010),'c.bmp']));
%% 生成方向场图
label = myresult; %prediction
label0 = trueresult; %true
rr = 1;
cell0 = cell(1,rr);
cell00 = cell(1,rr);
for l = 1:rr
    wa = reshape(label(l,:),20,20)';  %后面平滑滤波
    cell0{l} = wa;%cell0 存储label 1：400
    wb = reshape(label0(l,:),20,20)';
    cell00{l} = wb;
    wc = wa-wb;
    
end
[rows0,cols0] = size(haha); %160 160
result0 = zeros(rows0,cols0);
% result00 = zeros(rows0,cols0);

k0 = 1;
q0 = 1;
kk=1;
for kk = 1:rr
    n0=1;
    m0=1;
    k0=1;
for n0 = 1:8:rows0
    for m0 = 1:8:cols0  

        result0(n0:n0+15,m0:m0+15) = cell0{kk}(k0,q0)*ones(16)*pi/254;
%         result00(n0:n0+15,m0:m0+15) = cell00{kk}(k0,q0)*ones(16)*pi/254;
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
    plotridgeorient(result0, 8,haha, 1, 'r');
%     plotridgeorient(result00, 8,haha, 2, 'b');
    %end
    %show(reliability,6)
end