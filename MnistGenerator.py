#产生CGan训练集

import math
import numpy as np
from keras.datasets import mnist


class MnistGenerator(object):

    def __init__(self, batch_size, generator_noise_input_dim):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x = np.expand_dims((x_train.astype(np.float32) - 127.5) / 127.5, axis=-1)
        self.y = y_train.reshape(-1, 1)

        self.images_size = len(self.x)
        self.shuffle()

        self.generator_noise_input_dim = generator_noise_input_dim
        self.epoch = 1  # 当前迭代次数
        self.batch_num = math.ceil(len(self.x) / batch_size)
        self.batch_size = int(batch_size)
        self.start = 0
        self.end = 0
        self.finish_flag = False  # 遍历一遍标志

    def shuffle(self):
        """
        洗牌
        :return:
        """
        random_index = np.random.permutation(np.arange(self.images_size))
        self.x = self.x[random_index]
        self.y = self.y[random_index]

    def _next_batch(self):
        """
        返回一个batch
        :return:
        """
        while True:

            if self.finish_flag:  # 遍历一次
                self.shuffle()
                self.end = 0
                self.start = 0
                self.finish_flag = False
                self.epoch += 1

            self.end = int(np.min([self.images_size, self.start + self.batch_size]))
            batch_image = self.x[self.start:self.end]
            batch_label = self.y[self.start:self.end]
            batch_noise = np.random.normal(0,1, size=(self.end-self.start, self.generator_noise_input_dim))
            if self.end == self.images_size:
                self.finish_flag = True
            else:
                self.start = self.end
            yield batch_image, batch_label, batch_noise

    def next_batch(self):
        datagener = self._next_batch()
        return datagener.__next__()


if __name__ == '__main__':
    mg = MnistGenerator(40000, 100)
    for i in range(10):
        batch_image, batch_label, batch_noise = mg.next_batch()
        print(np.shape(batch_image), np.shape(batch_label), np.shape(batch_noise))
        #print(batch_image, batch_label)
        print(mg.epoch)