%% ��ȡlabel��output�ļ�
clear
clc
labelsize = 20; %label����Ĵ�С
num = 0; %��ȡ��ָ�ƺͶ�Ӧ��ǩ�ļ���

path = 'E:/fingerprint/1030/pic/'; %pics·��

% path1 = 'E:/fingerprint/1030/output'; %prediction���·��
% path2 =['E:/fingerprint/1030/labels/',int2str(num)]; %labels·��
% path = 'E:/fingerprint/1216/pic/'; %pics·��

path1 = 'E:\fingerprint\1030\labels\'; %prediction���·��

fp = fopen([path1,int2str(num),'.txt'],'r');
output = fscanf(fp,'%f',[1,inf]); 
fclose(fp);
myresult = round(output/127*254);
haha = double((imread([path,int2str(num),'.bmp'])));

%% ���ɷ���ͼ
label0 = myresult; %prediction
rr = 1;
cell00 = cell(1,rr);
for l = 1:rr
    cell00{l} = reshape(label0(l,:),20,20)'; %cell0 �洢my 1��400
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
    plotridgeorient(result00, 8,haha, 2, 'b');  %����������ԭͼ�ϵĿ��ӻ�Ч��ͼ

    %end
    %show(reliability,6)
end