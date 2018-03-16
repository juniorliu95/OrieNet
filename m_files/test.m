%% 对库指纹块数据集中三通道图像变为灰度图。
clear
clc
for i = 1:54446
% for i = 1:5
    a = imread(['E:\fingerprint\0921data\picku\',int2str(i-1),'.bmp']);
    if max(size(size(a)))==3
        a = rgb2gray(a);
        imwrite(a,['E:\fingerprint\0921data\picku1\',int2str(i-1),'.bmp']);
    else
        imwrite(a,['E:\fingerprint\0921data\picku1\',int2str(i-1),'.bmp']);
    end
    disp(i-1);
end