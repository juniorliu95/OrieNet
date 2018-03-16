%% for (0,pi) data
% clc
% fp = fopen('E:\fingerprint\0921data\doublelabel0921.txt','r');
% data = fscanf(fp,'%f',[1,inf]);
% fclose(fp);
% for i = 1:size(data,2)/100
%     disp(i);
%     temp = data((i-1)*100+1:i*100);
%     temp1 = zeros(10);
%     for j=1:100
%         temp1(j) = temp(j);
%     end
% %     temp1 = temp1';
%     fp = fopen(['E:\fingerprint\0921data\label\',int2str(i-1),'.txt'],'w+');
%     fprintf(fp,'%f ',temp1);
%     fclose(fp);
% end

%% for binary data
% clear
% clc
% fp = fopen('E:\fingerprint\0921data\5label0921.txt','r');
% data = fscanf(fp,'%f',[1,inf]);
% fclose(fp);
% data8 = zeros([size(data,2),8]);
% for i =1:size(data,2)
%     temp = zeros(1,8);
%     bin = dec2bin(data(i),8);
%     for j = 1:8
%         if bin(j) == '1'
%             temp(j) = 1;
%         end
%     end
%     data8(i,:) = temp;
%     disp(['this is ',int2str(i)]);
% end
% 
%  for i = 1:size(data,2)/100
% % for i = 1:2
%     temp = (data8((i-1)*100+1:i*100,:))';
%     temp1 = zeros(10,80);
%     for j=1:800
%         temp1(j) = temp(j);
%     end
% %     temp1 = temp1';
%     fp = fopen(['E:\fingerprint\0921data\binlabel\',int2str(i-1),'.txt'],'w+');
%     fprintf(fp,'%f ',temp1);
%     fclose(fp);
%     disp(['this is: ',int2str(i)]);
% end

%% for softmaxdata
clear
clc
fp = fopen('E:\fingerprint\0921data\5label0921.txt','r');
data = fscanf(fp,'%f',[1,inf]);
fclose(fp);
data256 = zeros([size(data,2),256]);
for i = 1:size(data,2)
    data256(i,data(i)+1) = 1;
end

 for i = 1:size(data,2)/100
% for i = 1:2
    temp = (data256((i-1)*100+1:i*100,:))';
    temp1 = zeros(10,2560);
    for j=1:25600
        temp1(j) = temp(j);
    end
%     temp1 = temp1';
    fp = fopen(['E:\fingerprint\0921data\slabel\',int2str(i-1),'.txt'],'w+');
    fprintf(fp,'%f ',temp1);
    fclose(fp);
    disp(['this is: ',int2str(i)]);
end