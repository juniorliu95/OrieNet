%% 将有效区域8倍降采样
clear
clc

path = 'E:\fingerprint\0117\';
path1 = [path,'mask\'];
path2 = [path,'mask1\'];
path3 = [path,'mask01\'];
% for i = 1: 100000
% % for i = 1
%     disp(i);
%     img = imread([path1,int2str(i-1),'.bmp']);
% %     imshow(img);
%     rows = size(img,1);
%     cols = size(img,2);
%     img1 = uint8(zeros(rows/8,cols/8));
%     for j=1:rows/8
%         for k = 1:cols/8
%             if min(min(img((j-1)*8+1:j*8,(k-1)*8+1:k*8))) ==0
%                 img1(j,k)=0;
%             else
%                 img1(j,k)=255;
%             end
%         end
%     end
% %     figure,imshow(img1);
%     imwrite(img1,[path2,int2str(i-1),'.bmp']);
% end

%% 将mask变为0,1 uint8
for i = 1: 100000
% for i = 1
    disp(i);
    img = imread([path1,int2str(i-1),'.bmp']);
%     imshow(img);
    rows = size(img,1);
    cols = size(img,2);
    img1 = uint8(double(img)/255);

%     figure,imshow(img1,[]);
    imwrite(img1,[path3,int2str(i-1),'.bmp']);
end
