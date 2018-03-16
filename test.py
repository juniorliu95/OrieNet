import tensorflow as tf
import numpy as np
import fingerprit
import utils
import cv2
import os
from PIL import Image

np.set_printoptions(threshold=np.inf)
# path = 'E:/fingerprint/1030/'
# path = '/home/higo/DataSet/0921data/'
# path = 'E:/fingerprint/1216/'
# path = 'E:/fingerprint/1222/'
# path = 'E:/fingerprint/1224/'
path = './'# path of .npy and 0112pic file
# path = 'E:/fingerprint/0102/'
batch_size = 1
batches = tf.placeholder(tf.float32, [None, 512, 512, 1])
train_mode = tf.placeholder(tf.bool)
vgg = fingerprit.FingerPrint(path+'test3-save.npy')
vgg.build(batches, batch_size, train_mode)

# img = cv2.imread(path+"pic/113.bmp", 0)
# with tf.device('gpu:0'):
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for num1 in range(1, 31):  # 测试文件夹名称
#             pic_path = path + "test/" + str(num1) + "/"
#             num = len(os.listdir(pic_path))  # 图的数量
#             print('\n')
#             print(num1)
#             print(num)
#             print('\n')
#             for k in range(num):
#                 print(k)
#                 img = cv2.imread(path+"test/"+str(num1)+"/" + str(k) + ".png", 0)
#                 document = open(path+"test/"+str(num1)+"/" + str(k) + ".txt", "w+")
#                 img1 = cv2.resize(img, (160, 160)).reshape((1, 160, 160, 1))
#                 mean = np.mean(img1)
#                 stddev = np.sqrt(np.mean((img1 - mean) ** 2))
#                 img1 = (img1 - mean) / stddev
#                 # print(img1)
#                 # img1 = img1 * (1./255) - 0.5
#                 # img2 = tf.Variable(img1, dtype=tf.float32)
#                 # document = open(path+"output.txt", "w+")
#                 # document = open(path + num + ".txt", "w+")
#                 # print(sess.run(img2))
#                 output = sess.run(vgg.getoutput(vgg.fc9), feed_dict={batches: img1, train_mode: False})
#                 output = np.reshape(output, [20, 20])
#                 # print(output)
#                 for i in range(20):
#                     for j in range(20):
#                 # document.write(str(output[:]))
#                         document.write(str(int(output[i][j])))
#                         document.write('\n')
#                 document.close()

# 整张图测试
with tf.device('gpu:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i, name in enumerate(os.listdir(path+"0112pic/")):
            print(path + name)
            img = cv2.imread(path + "0112pic/" + name, 0)
            document = open(path + "0112rs/" + name[:-4] + ".txt", "w+")
            img1 = cv2.resize(img, (512, 512)).reshape((1, 512, 512, 1))
            mean = np.mean(img1)
            stddev = np.sqrt(np.mean((img1 - mean) ** 2))
            img1 = (img1 - mean) / stddev
            # print(img1)
            # img1 = img1 * (1./255) - 0.5
            # img2 = tf.Variable(img1, dtype=tf.float32)
            # document = open(path+"output.txt", "w+")
            # document = open(path + num + ".txt", "w+")
            # print(sess.run(img2))
            output = sess.run(vgg.getoutput(vgg.fc9), feed_dict={batches: img1, train_mode: False})
            output = np.reshape(output, [64, 64])
            # print(output)
            for i in range(64):
                for j in range(64):
            # document.write(str(output[:]))
                    document.write(str(int(output[i][j])))
                    document.write('\n')
            document.close()


