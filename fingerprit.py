"""
during training process:
input: 160*160
output: 20*20 = 400
total: *8 downsampling
"""
import tensorflow as tf
import numpy as np
from functools import reduce

class FingerPrint:

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None
            # dictionary to store all the data of the network

        self.var_dict = {}
        # dictionary to store the value of all the variable
        # format: self.var_dict[(name, idx)] = var
        self.trainable = trainable
        self.dropout = dropout


    def build(self, image, batch_size, train_mode=True):
        """
        network structure
        """
        self.train_mode = train_mode
        # assert image.shape[1:] == [160, 160, 1]
        self.conv1_11 = self.conv_layer(image, 1, 64, "conv1_11", conv_size=7)
        self.conv1_21 = self.conv_layer(self.conv1_11, 64, 128, "conv1_21", conv_size=7, stride=2)
        self.relu11 = tf.nn.relu(self.conv1_21)
        # self.bn1 = self.batch_norm_layer(self.relu1, name='bn1')
        # self.pool1 = self.max_pool(self.relu1, 'pool1')
        # self.conv1_12 = self.conv_layer(image, 1, 64, "conv1_12", conv_size=7)
        # self.conv1_22 = self.conv_layer(self.conv1_12, 64, 128, "conv1_22", conv_size=7, stride=2)
        # self.relu12 = tf.nn.relu(self.conv1_22)
        # # self.bn1 = self.batch_norm_layer(self.relu1, name='bn1')
        # # self.pool1 = self.max_pool(self.relu1, 'pool1')
        # self.conv1_13 = self.conv_layer(image, 1, 64, "conv1_13", conv_size=7)
        # self.conv1_23 = self.conv_layer(self.conv1_13, 64, 128, "conv1_23", conv_size=7, stride=2)
        # self.relu13 = tf.nn.relu(self.conv1_23)
        # # self.bn1 = self.batch_norm_layer(self.relu1, name='bn1')
        # # self.pool1 = self.max_pool(self.relu1, 'pool1')

        self.conv2_11 = self.conv_layer(self.relu11, 128, 256, "conv2_11", conv_size=5, stddev=0.001)
        self.conv2_21 = self.conv_layer(self.conv2_11, 256, 256, "conv2_21", conv_size=5, stride=2)
        self.relu21 = tf.nn.relu(self.conv2_21)
        # self.bn2 = self.batch_norm_layer(self.relu2,name='bn2')
        # self.pool2 = self.max_pool(self.relu2, 'pool2')
        # self.conv2_12 = self.conv_layer(self.relu12, 128, 256, "conv2_12", conv_size=5, stddev=0.001)
        # self.conv2_22 = self.conv_layer(self.conv2_12, 256, 256, "conv2_22", conv_size=5, stride=2)
        # self.relu22 = tf.nn.relu(self.conv2_22)
        # # self.bn2 = self.batch_norm_layer(self.relu2,name='bn2')
        # # self.pool2 = self.max_pool(self.relu2, 'pool2')
        # self.conv2_13 = self.conv_layer(self.relu13, 128, 256, "conv2_13", conv_size=5, stddev=0.001)
        # self.conv2_23 = self.conv_layer(self.conv2_13, 256, 256, "conv2_23", conv_size=5, stride=2)
        # self.relu23 = tf.nn.relu(self.conv2_23)
        # self.bn2 = self.batch_norm_layer(self.relu2,name='bn2')
        # self.pool2 = self.max_pool(self.relu2, 'pool2')

        self.conv3_11 = self.conv_layer(self.relu21, 256, 512, "conv3_11", conv_size=3)
        self.conv3_21 = self.conv_layer(self.conv3_11, 512, 512, "conv3_21", conv_size=3, stride=2)  # 8-times downsampling
#        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        # self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.relu31 = tf.nn.relu(self.conv3_21)
        # self.bn3 = self.batch_norm_layer(self.relu3, name='bn3')
        # self.pool3 = self.max_pool(self.relu3, 'pool3')
        # self.conv3_12 = self.conv_layer(self.relu22, 256, 512, "conv3_12", conv_size=3)
        # self.conv3_22 = self.conv_layer(self.conv3_12, 512, 512, "conv3_22", conv_size=3, stride=2)  # 8-times downsampling
        # #        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        # # self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        # self.relu32 = tf.nn.relu(self.conv3_22)
        # # self.bn3 = self.batch_norm_layer(self.relu3, name='bn3')
        # # self.pool3 = self.max_pool(self.relu3, 'pool3')
        # self.conv3_13 = self.conv_layer(self.relu23, 256, 512, "conv3_13", conv_size=3)
        # self.conv3_23 = self.conv_layer(self.conv3_13, 512, 512, "conv3_23", conv_size=3, stride=2)  # 8-times downsampling
        # #        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        # # self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        # self.relu33 = tf.nn.relu(self.conv3_23)
        # self.bn3 = self.batch_norm_layer(self.relu3, name='bn3')
        # self.pool3 = self.max_pool(self.relu3, 'pool3')

        # multiple layers rate=2,4,6.
        self.conv4_11 = self.atrous_layer(self.relu31, 512, 256, "conv4_11", rate=1)
        self.conv4_12 = self.conv_layer(self.conv4_11, 256, 256, "conv4_12")
        self.relu411 = tf.nn.relu(self.conv4_12)
        self.conv4_13 = self.atrous_layer(self.relu31, 512, 256, "conv4_13", rate=2)
        self.conv4_14 = self.atrous_layer(self.conv4_13, 256, 256, "conv4_14", rate=2)
        self.relu412 = tf.nn.relu(self.conv4_14)
        # self.conv4_13 = self.atrous_layer(self.relu31, 512, 256, "conv4_13", rate=6)
        # self.relu413 = tf.nn.relu(self.conv4_13)
        # self.pool4 = self.max_pool(self.relu4, 'pool4')
        self.conv5_1 = self.conv_layer(0.7 * self.relu411 + 0.3 * self.relu412, 256, 256, "conv5_1", conv_size=1)

        # multiple layers rate=2,4,6.
        self.conv4_21 = self.atrous_layer(self.relu31, 512, 256, "conv4_21", rate=1)
        self.conv4_22 = self.conv_layer(self.conv4_21, 256, 256, "conv4_22")
        self.relu421 = tf.nn.relu(self.conv4_22)
        self.conv4_23 = self.atrous_layer(self.relu31, 512, 256, "conv4_23", rate=2)
        self.conv4_24 = self.conv_layer(self.conv4_23, 256, 256, "conv4_24")
        self.relu422 = tf.nn.relu(self.conv4_24)
        # self.conv4_23 = self.atrous_layer(self.relu32, 512, 256, "conv4_23", rate=6)
        # self.relu423 = tf.nn.relu(self.conv4_23)
        # self.pool4 = self.max_pool(self.relu4, 'pool4')
        self.conv5_2 = self.conv_layer(0.7 * self.relu421 + 0.3 * self.relu422, 256, 256, "conv5_2", conv_size=1)

        # multiple layers rate=2,4,6.
        self.conv4_31 = self.atrous_layer(self.relu31, 512, 256, "conv4_31", rate=1)
        self.conv4_32 = self.conv_layer(self.conv4_31, 256, 256, "conv4_32")
        self.relu431 = tf.nn.relu(self.conv4_32)
        self.conv4_33 = self.atrous_layer(self.relu31, 512, 256, "conv4_33", rate=2)
        self.conv4_34 = self.conv_layer(self.conv4_33, 256, 256, "conv4_32")
        self.relu432 = tf.nn.relu(self.conv4_34)
        # self.conv4_33 = self.atrous_layer(self.relu33, 512, 256, "conv4_33", rate=6)
        # self.relu433 = tf.nn.relu(self.conv4_33)
        # self.pool4 = self.max_pool(self.relu4, 'pool4')
        self.conv5_3 = self.conv_layer(0.7 * self.relu431 + 0.3 * self.relu432, 256, 256, "conv5_3", conv_size=1)

        self.fc81 = self.conv_layer(self.conv5_1, 256, 1, "fc8_1", conv_size=1)
        self.fc82 = self.conv_layer(self.conv5_2, 256, 1, "fc8_2", conv_size=1)
        self.fc83 = self.conv_layer(self.conv5_3, 256, 1, "fc8_3", conv_size=1)
        self.fc9 = tf.concat(axis=3, values=[self.fc81, self.fc82, self.fc83])  # 三层输出，每层都在(0,0.1)

    def getoutput(self, output):  # 得到转换前的三层值，范围为[0, 179],并使用boosting得到最后的角度预测输出
        change = tf.round(tf.multiply(output, 1790))
        final = change
        change1 = tf.nn.relu(change)  # 负值变为0
        change = 179 - tf.nn.relu(179 - change1)  #大于179的值变为179

        # 变换到(0,180)范围内
        layer1, layer2, layer3 = tf.split(change, 3, -1)
        # zeros = tf.zeros(tf.shape(layer1))
        layer21 = tf.nn.relu(layer2 - 120)  # 取出（120,179），变为(0,59)
        layer22 = 119 - tf.nn.relu(119 - layer2)  # 取出（0,119），其余变为119
        temp2 = tf.cast(tf.equal(layer22, layer2), tf.float32)  # 变为119的位置输出为0
        layer2 = layer21 + (layer22 + 60) * temp2
        layer31 = tf.nn.relu(layer3 - 60)  # 取出（60,179），变为（0,119）
        layer32 = 59 - tf.nn.relu(59 - layer3)  # 取出（0,59），其余变为59
        temp3 = tf.cast(tf.equal(layer32, layer3), tf.float32)
        layer3 = layer31 + (layer32 + 120) * temp3
        final = tf.concat(axis=3, values=[layer1, layer2, layer3])

        # 得到boosting选择后的输出
        layer4 = tf.abs(layer2 - layer1)  # 查layer1和2的误差
        layer41 = tf.nn.relu(layer4 - 15) + 15  # 小于x的输出全部变成x
        temp4 = tf.cast(tf.equal(layer41, layer4), tf.float32)  # 小于x的位置输出为0，大于x为1
        final = tf.floor((layer1 + layer2) / 2) * (1 - temp4) + layer3 * temp4

        return final

    # def batch_norm_layer(self, x, name):
    #     with tf.variable_scope(name+'_bn'):
    #         beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
    #         gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
    #         axises = list(range(len(x.shape) - 1))  # [0 1 2]
    #         batch_mean, batch_var = tf.nn.moments(x, axises)  # 得到输入层的mean和var
    #         ema = tf.train.ExponentialMovingAverage(decay=0.5)  # 衰减系数0.5
    #         # print(self.train_mode)
    #
    #         def mean_var_with_update():
    #             ema_apply_op = ema.apply([batch_mean, batch_var])  # 保持更新的变量组
    #             with tf.control_dependencies([ema_apply_op]):  # 下文必须在ema_apply_op完成后完成
    #                 return tf.identity(batch_mean), tf.identity(batch_var)  # 返回的始终为更新后的值
    #
    #         # mean, var = mean_var_with_update()
    #         mean, var = tf.cond(self.train_mode,
    #                             lambda: mean_var_with_update(),
    #                             lambda: (ema.average(batch_mean), ema.average(batch_var))
    #                             )
    #         normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    #     return normed

    def avg_pool(self, bottom, name, stride=2):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name, stride=2):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name, padding='SAME', conv_size=3, stddev=0.001, stride=1):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(conv_size, in_channels, out_channels, name, stddev)
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding=padding)
            bias = tf.nn.bias_add(conv, conv_biases)
            # relu = tf.nn.relu(bias)
            return bias

    def atrous_layer(self, bottom, in_channels, out_channels, name, padding='SAME', rate=2, stddev=0.001):  # 空洞卷积
        filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name, stddev)
        conv = tf.nn.atrous_conv2d(bottom, filt, rate, padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return bias

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name, stddev):  # name = 'convx_x'
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, stddev)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, stddev)  # shape, mean, stddev
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value
            print(name + ' value initial')

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        # print(var.get_shape(), initial_value.get_shape())
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./test2-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            # format of var_dict.items(): dict_items([((name1, idx1), var1), ((name2, idx2), var2)])
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count


    #read tfrecord file
    def read_and_decode(self, filename, epoch=None):
        filename_queue = tf.train.string_input_producer\
            ([filename], num_epochs=epoch, shuffle=True)#生成一个random queue队列
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)#返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.string),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })  # 将image数据和label取出来

        img0 = tf.decode_raw(features['img_raw'], tf.uint8)
        img0 = tf.reshape(img0, [160, 160, 1])  # reshape为128*128的1通道图片
        img0 = tf.cast(img0, tf.float32)
        mean = tf.reduce_mean(img0)
        std = tf.sqrt(tf.reduce_mean((img0-mean)**2))
        img0 = (tf.cast(img0, tf.float32) - mean) * (1./std)  # 白化
        label = tf.decode_raw(features['label'], tf.float64)  # 在流中抛出label张量
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [20, 20, 3])
        label = label * (1./1790)  # 变换到（-0.1，0）
        return img0, label