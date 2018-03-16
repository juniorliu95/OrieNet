%% 把labels变为以0,60,120°为原点的三组

clear
clc
% path = 'E:\fingerprint\1030\labels\';
% outpath = 'E:\fingerprint\1030\labels1\';
path = 'E:\fingerprint\1228\labels\';
outpath = 'E:\fingerprint\1228\labels1\';
for i = 1:100000
% for i = 1
    disp(i-1);
    
    fp = fopen([path, int2str(i-1), '.txt'], 'r');
    data = round(fscanf(fp,'%f',[1,inf])/127*179);
    fclose(fp);
    a = reshape(data,[20,20]); %以0,180作为原点
    b = reshape(data,[20,20]); %以60作为原点
    c = reshape(data,[20,20]); %以120作为原点
    d = zeros(3,20,20); 
    for j = 1:400
%             if( a(j)>90 )
%                 a(j) = a(j)-180;
%             end
            if( b(j)<60 )
                b(j) = b(j)+120;
            else
                b(j) = b(j)-60;
            end
            if( c(j)<120 )
                c(j) = c(j)+60;
            else
                c(j) = c(j)-120;
            end
    end
    d(1,:,:) = a;
    d(2,:,:) = b;
    d(3,:,:) = c;
    fp = fopen([outpath, int2str(i-1), '.txt'], 'w+');
    fprintf(fp,'%d ',d);
    fclose(fp);
    
end

%% 变到2层 输出为0到90 带正负
% clear
% clc
% path = 'E:\fingerprint\1030\labels\';
% outpath = 'E:\fingerprint\1030\labels2\';
% % for i = 1:109893
% for i = 1
%     disp(i-1);
%     
%     fp = fopen([path, int2str(i-1), '.txt'], 'r');
%     data = fscanf(fp,'%f',[1,inf]);
%     fclose(fp);
%     a = abs(reshape(data,[20,20])-64);
%     b = reshape(data,[20,20])-64;
%     for j =1:400
%         if b(j)<0
%             b(j) = 0;
%         else
%             b(j) = 1;
%         end
%     end
%     d = zeros(2,20,20);
%     d(1,:,:) = a;
%     d(2,:,:) = b;
%     fp = fopen([outpath, int2str(i-1), '.txt'], 'w+');
%     fprintf(fp,'%d ',d);
%     fclose(fp);
%     
% end