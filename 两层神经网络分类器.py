# 导入需要的包
import random
import pandas as pd
import numpy
import scipy.special
import os
import dill
import matplotlib.pyplot as plt

# 两层神经网络分类器类定义
class two_layers_neural_network:

    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes):
        # 设置每个输入、隐藏、输出层的节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 权值矩阵和偏置初始化
        self.w1 = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.w2 = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.b1 = numpy.zeros((self.hnodes, 1))
        self.b2 = numpy.zeros((self.onodes, 1))

        # 激活函数是sigmoid型函数
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # 查询神经网络
    def query(self, inputs_list):
        # 将输入列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 计算信号到隐藏层
        hidden_inputs = numpy.dot(self.w1, inputs) + self.b1
        # 计算从隐含层出现的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算信号到最终的输出层
        final_inputs = numpy.dot(self.w2, hidden_outputs) + self.b2
        # 计算从最终输出层出现的信号
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # 求loss
    def loss(self, inputs, labels, Lambda):
        targets = numpy.zeros((10, labels.shape[0])) + 0.01
        for i in range(labels.shape[0]):
            targets[int(labels[i, 0]), i] = 0.99
        T = targets - self.query(inputs)
        loss_ = (numpy.sum(T * T) + Lambda*(numpy.sum(self.w1 * self.w1) + numpy.sum(self.w2 * self.w2))) / 2
        return loss_

    # 用一条数据训练神经网络
    def train1(self, inputs_list, targets_list, lr, Lambda):
        # 将输入列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算信号到隐藏层
        hidden_inputs = numpy.dot(self.w1, inputs) + self.b1
        # 计算从隐含层出现的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算信号到最终的输出层
        final_inputs = numpy.dot(self.w2, hidden_outputs) + self.b2
        # 计算从最终输出层出现的信号
        final_outputs = self.activation_function(final_inputs)

        # 输出层误差为(目标值-实际值)
        output_errors = targets - final_outputs
        # 隐藏层错误是output_errors，按权重分割，在隐藏节点处重新组合
        hidden_errors = numpy.dot(self.w2.T, output_errors)

        # 更新隐藏层和输出层之间的偏置
        self.b2 += lr * output_errors * final_outputs * (1.0 - final_outputs)

        # 更新隐藏层和输入层之间的偏置
        self.b1 += lr * hidden_errors * hidden_outputs * (1.0 - hidden_outputs)

        # 更新隐藏层和输出层之间的链接的权重
        self.w2 += lr * (numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs)) - Lambda * self.w2)

        # 更新输入层和隐藏层之间的链接的权值
        self.w1 += lr * (numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs)) - Lambda * self.w1)
        pass

    # 采取指数型学习率下降策略和SGD方法训练
    def train(self, inputs, targets, lr, Lambda, epochs=10, batch=100, decay_rate=0.99):
        k = int(inputs.shape[0] / batch)
        for e in range(epochs):
            # 打乱数据顺序
            id = numpy.arange(inputs.shape[0])
            random.shuffle(id)
            for t in range(k):
                lr = lr * decay_rate
                for i in range(batch):
                    # 规模和转移输入
                    input = inputs[id[t * batch + i], :]
                    # 创建目标输出值(都是0.01，除了所需的标签为0.99)
                    target = numpy.zeros((1, self.onodes)) + 0.01
                    target[0, int(targets[id[t * batch + i], 0])] = 0.99
                    self.train1(input, target, lr, Lambda)
        pass

    # 计算准确度
    def acc(self, inputs, targets):
        label = numpy.zeros(targets.shape)
        for i in range(inputs.shape[0]):
            input = inputs[i, ]
            label[i, 0] = numpy.argmax(self.query(input))
        return numpy.sum(label == targets) / inputs.shape[0]

    # 绘图
    def train_best(self, inputs, targets, validation_x, validation_y, lr, Lambda, epochs=10, batch=100, decay_rate=0.99):
        losslist1 = []
        losslist2 = []
        acclist = []
        k = int(inputs.shape[0] / batch)
        for e in range(epochs):
            # 打乱数据顺序
            id = numpy.arange(inputs.shape[0])
            random.shuffle(id)
            for t in range(k):
                lr = lr * decay_rate
                for i in range(batch):
                    # 规模和转移输入
                    input = inputs[id[t * batch + i], :]
                    # 创建目标输出值(都是0.01，除了所需的标签为0.99)
                    target = numpy.zeros((1, self.onodes)) + 0.01
                    target[0, int(targets[id[t * batch + i], 0])] = 0.99
                    self.train1(input, target, lr, Lambda)
                if t % 5 == 0:
                    losslist1.append(self.loss(inputs, targets, Lambda))
                    losslist2.append(self.loss(validation_x, validation_y, Lambda))
                    acclist.append(self.acc(validation_x, validation_y))
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.grid(True)
        list = numpy.arange(5, 5 * (len(losslist1) + 1), 5)
        plt.plot(list, losslist1, c='blue', marker='o', linestyle=':', label='train_loss')
        plt.plot(list, losslist2, c='red', marker='*', linestyle='-', label='validation_loss')
        plt.xlabel('批次')
        plt.ylabel('loss')
        plt.title('训练过程中loss变化')
        plt.legend(['train_loss', 'validation_loss'])
        plt.savefig('loss.png')
        plt.show()
        plt.figure()
        plt.grid(True)
        plt.plot(list, acclist)
        plt.xlabel('批次')
        plt.ylabel('accuracy')
        plt.title('训练过程中accuracy变化')
        plt.savefig('acc.png')
        plt.show()
    # 验证集上寻找较优参数
    def best_para(self, inputs, targets, validation_inputs, validation_targets, lrlist, lambdalist, hnodelist):
        acc = 0
        for lr in lrlist:
            for Lambda in lambdalist:
                for hnode in hnodelist:
                    self.hnodes = hnode
                    self.w1 = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
                    self.w2 = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
                    self.b1 = numpy.zeros((self.hnodes, 1))
                    self.b2 = numpy.zeros((self.onodes, 1))
                    self.train(inputs, targets, lr, Lambda)
                    acc_ = self.acc(validation_inputs, validation_targets)
                    print('学习率，隐藏层大小，正则化强度分别为{0},{1},{2}时的分类精度为：'.format(lr, hnode, Lambda),
                          acc_)
                    if acc_ > acc:
                        acc = acc_
                        w1 = self.w1
                        w2 = self.w2
                        b1 = self.b1
                        b2 = self.b2
                        bestlr = lr
                        besthnode = hnode
                        bestLambda = Lambda

        # 绘图
        self.hnodes = besthnode
        self.w1 = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.w2 = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.b1 = numpy.zeros((self.hnodes, 1))
        self.b2 = numpy.zeros((self.onodes, 1))
        self.train_best(inputs, targets, validation_inputs, validation_targets, bestlr, bestLambda)
        # 保存模型
        if acc > 0.9:
            with open('model.pkl', 'wb') as fp:
                dill.dump(self, fp, -1)
        else:
            print("警告：验证集上准确率仅为", acc)


if __name__ == '__main__':
    # 读入数据
    train = pd.read_csv(r'E:\课程\深度学习\Handwritten_digit_recognition\MNIST_data\mnist_train.csv')
    test = pd.read_csv(r'E:\课程\深度学习\Handwritten_digit_recognition\MNIST_data\mnist_test.csv')
    # 分割训练集、测试集、验证集
    train_x = (train.iloc[0: 50000, 1:].values / 255.0 * 0.99) + 0.01
    train_y = train.iloc[0: 50000, 0].values.reshape(50000, 1)
    validation_x = (train.iloc[50000: 59999, 1:].values / 255.0 * 0.99) + 0.01
    validation_y = train.iloc[50000: 59999, 0].values.reshape(9999, 1)
    test_x = (test.iloc[:, 1:].values / 255.0 * 0.99) + 0.01
    test_y = test.iloc[:, 0].values.reshape(9999, 1)

    # 生成网络
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as fp:
            network = dill.load(fp)
        print("模型在测试集上的准确率为：", network.acc(test_x, test_y))
    else:
        network = two_layers_neural_network(784, 50, 10)
        # 训练神经网络
        lrlist = [0.1, 0.05, 0.01]
        Nlist = [100, 50, 30]
        Lambdalist = [0, 0.1, 0.01]
        network.best_para(train_x, train_y, validation_x, validation_y, lrlist, Lambdalist, Nlist)
        print("模型在测试集上的准确率为：", network.acc(test_x, test_y))

    # 可视化权值矩阵
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.imshow(network.w1, cmap='magma')
    plt.colorbar()
    plt.title('输入层到隐藏层的权值矩阵热度图')
    plt.savefig('w1.png')
    plt.show()
    plt.figure()
    plt.imshow(network.w2, cmap='magma')
    plt.colorbar()
    plt.title('隐藏层到输出层的权值矩阵热度图')
    plt.savefig('w2.png')
    plt.show()


'''
输出结果为：
学习率，隐藏层大小，正则化强度分别为0.1,100,0时的分类精度为： 0.9326932693269326
学习率，隐藏层大小，正则化强度分别为0.1,50,0时的分类精度为： 0.9287928792879288
学习率，隐藏层大小，正则化强度分别为0.1,30,0时的分类精度为： 0.9214921492149215
学习率，隐藏层大小，正则化强度分别为0.1,100,0.1时的分类精度为： 0.10641064106410641
学习率，隐藏层大小，正则化强度分别为0.1,50,0.1时的分类精度为： 0.10641064106410641
学习率，隐藏层大小，正则化强度分别为0.1,30,0.1时的分类精度为： 0.10641064106410641
学习率，隐藏层大小，正则化强度分别为0.1,100,0.01时的分类精度为： 0.845084508450845
学习率，隐藏层大小，正则化强度分别为0.1,50,0.01时的分类精度为： 0.8191819181918192
学习率，隐藏层大小，正则化强度分别为0.1,30,0.01时的分类精度为： 0.748974897489749
学习率，隐藏层大小，正则化强度分别为0.05,100,0时的分类精度为： 0.9183918391839184
学习率，隐藏层大小，正则化强度分别为0.05,50,0时的分类精度为： 0.9158915891589159
学习率，隐藏层大小，正则化强度分别为0.05,30,0时的分类精度为： 0.90999099909991
学习率，隐藏层大小，正则化强度分别为0.05,100,0.1时的分类精度为： 0.10641064106410641
学习率，隐藏层大小，正则化强度分别为0.05,50,0.1时的分类精度为： 0.10641064106410641
学习率，隐藏层大小，正则化强度分别为0.05,30,0.1时的分类精度为： 0.10641064106410641
学习率，隐藏层大小，正则化强度分别为0.05,100,0.01时的分类精度为： 0.8308830883088308
学习率，隐藏层大小，正则化强度分别为0.05,50,0.01时的分类精度为： 0.7688768876887688
学习率，隐藏层大小，正则化强度分别为0.05,30,0.01时的分类精度为： 0.7451745174517451
学习率，隐藏层大小，正则化强度分别为0.01,100,0时的分类精度为： 0.861986198619862
学习率，隐藏层大小，正则化强度分别为0.01,50,0时的分类精度为： 0.8312831283128312
学习率，隐藏层大小，正则化强度分别为0.01,30,0时的分类精度为： 0.8523852385238524
学习率，隐藏层大小，正则化强度分别为0.01,100,0.1时的分类精度为： 0.10641064106410641
学习率，隐藏层大小，正则化强度分别为0.01,50,0.1时的分类精度为： 0.10641064106410641
学习率，隐藏层大小，正则化强度分别为0.01,30,0.1时的分类精度为： 0.10641064106410641
学习率，隐藏层大小，正则化强度分别为0.01,100,0.01时的分类精度为： 0.7653765376537653
学习率，隐藏层大小，正则化强度分别为0.01,50,0.01时的分类精度为： 0.7116711671167116
学习率，隐藏层大小，正则化强度分别为0.01,30,0.01时的分类精度为： 0.5936593659365936
模型在测试集上的准确率为： 0.9274927492749275
'''