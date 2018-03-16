a = reshape(output2',1,812);
aa = a(:,1:800);
result = reshape(aa,8,100)';
for i = 1:800
if result(i)>1
result(i) = 1;
end
end

r = result;
myresult = zeros(1,100);

for i = 1:100
       myresult(i) = 2^7*r(i,1)+2^6*r(i,2)+2^5*r(i,3)+2^4*r(i,4)+2^3*r(i,5)+2^2*r(i,6)+2^1*r(i,7)+2^0*r(i,8);
end

%clear
%%
%查看聚类获得的标签方向场数据
haha = double(rgb2gray(imread('looklabel.png')));
%fid = fopen('result100first.txt','r');
%fid = fopen('22578.txt','r');

%[data,count] = fread(fid,inf,'double');
%xx = reshape(data,100,22578)';
%tic;

%%%%
%[label0,center] = litekmeans(xx,128,'MaxIter',20);
%label0 = int32(label0);
%toc;
%%尝试分类
%strp = 'E:\\0922\\';
%for p = 0:127
%    fj = strcat(strp,num2str(p));
%    mkdir(fj);
%end

%for o = 0:22577
%    output = strcat('E:\\0922\\',num2str(label0(o+1)-1));
%    picture = strcat(num2str(o),'.png');
%    temp = strcat(output,'\\');
%    outputpath = strcat(temp,picture);
%    path = strcat('E:\\copy0824intense\\',picture);
%    copyfile(path,outputpath);
%end

%%
%label = load('result100first.txt');
label = myresult;

rr = 1;
cell0 = cell(1,rr);
for l = 1:rr
    cell0{l} = reshape(label(l,:),10,10)';
end
[rows0,cols0] = size(haha);
result0 = zeros(rows0,cols0);

k0 = 1;
q0 = 1;
kk=1;
for kk = 1:rr
    n0=1;
    m0=1;
    k0=1;
for n0 = 1:16:rows0
    for m0 = 1:16:cols0  

        result0(n0:n0+15,m0:m0+15) = cell0{kk}(k0,q0)*ones(16)*pi/254;
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
    plotridgeorient(result0, 16,haha, 2);
    %end
    %show(reliability,6)
end

 

