%% 合并../test文件夹的测试图像数据
clear
clc

a= imread('E:\fingerprint\0921data\test\11010a.jpg');
b = imread('E:\fingerprint\0921data\test\11010b.jpg');
c = imread('E:\fingerprint\0921data\test\11010c.jpg');

y1 = 7;
x1 = 10;
y2 = 10;
x2 = 7;
y3 = 10;
x3 = 16;
out = uint8(zeros([32*16,32*16,3]));
for i = 1:3
    matrix = uint8(zeros(32*16));
    matrix(x1*16+1:x1*16+160,y1*16+1:y1*16+160) = a(:,:,i);
%     figure,imshow(matrix);
    matrix(x2*16+1:x2*16+160,y2*16+1:y2*16+160) = b(:,:,i);
%     figure,imshow(matrix);
    matrix(x3*16+1:x3*16+160,y3*16+1:y3*16+160) = c(:,:,i);
    figure,imshow(matrix);
    out(:,:,i) = matrix;
end
imshow(out);