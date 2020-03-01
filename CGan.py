
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from keras import Input, Model, Sequential
from keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Dropout, Flatten, multiply, Embedding
from keras.layers import Conv2D, Conv2DTranspose, Activation
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from copy import deepcopy
from keras.utils import plot_model



class CGan(object):


    def __init__(self, config, weight_path = None):
        """
        CGan初始化函数
        :param config:配置文件
        :param weight_path: 已有权重路径
        """
        self.config = config
        self.build_cgan_model()
        if weight_path is not None:
            self.cgan.load_weights(weight_path, by_name = True)


    def build_cgan_model(self):
        """
        build cgan model
        :return:
        """
        #初始化输入
        self.generator_noise_input = Input(shape=(self.config.generator_noise_input_dim,))
        self.discriminator_image_input = Input(shape=self.config.discriminator_image_input_dim)
        self.contational_label_input = Input(shape=(1,), dtype='int32')

        #定义优化器
        self.optimizer = Adam(lr=2e-4, beta_1=0.5)

        #构建生成器模型与判别器模型
        self.discriminator_model = self.build_discriminator_model()
        self.discriminator_model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.generator_model = self.build_generator()

        #构建CGan
        self.discriminator_model.trainable = False
        self.cgan_input = [self.generator_noise_input, self.contational_label_input]
        generator_output = self.generator_model(self.cgan_input)

        self.discriminator_input = [generator_output, self.contational_label_input]
        self.cgan_output = self.discriminator_model(self.discriminator_input)
        self.cgan = Model(self.cgan_input, self.cgan_output)

        self.cgan.compile(optimizer=self.optimizer, loss='binary_crossentropy')
        plot_model(self.cgan, "./model/CDCGan_Model.png")
        plot_model(self.generator_model, "./model/CDCGan_generator.png")
        plot_model(self.discriminator_model, "./model/CDCGan_discriminator.png")


    def build_discriminator_model(self):
        """

        :return:
        """
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same',
                         input_shape=self.config.discriminator_image_input_dim))
        model.add(LeakyReLU(alpha=self.config.LeakyReLU_alpha))
        model.add(Dropout(rate=self.config.dropout_prob))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=self.config.LeakyReLU_alpha))
        model.add(Dropout(rate=self.config.dropout_prob))

        model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=self.config.LeakyReLU_alpha))
        model.add(Dropout(rate=self.config.dropout_prob))

        model.add(Conv2D(512, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=self.config.LeakyReLU_alpha))
        model.add(Dropout(rate=self.config.dropout_prob))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.config.discriminator_image_input_dim)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = (Embedding(self.config.condational_label_num,
                                              np.prod(self.config.discriminator_image_input_dim))(label))

        label_embedding = Reshape(self.config.discriminator_image_input_dim)(label_embedding)
        model_input = multiply([img, label_embedding])
        validity = model(model_input)

        return Model([img, label], validity)


    def build_generator(self):
        """
        这是构建生成器网络的函数
        :return:返回生成器模型generotor_model
        """
        model = Sequential()

        model.add(Dense(7*7*256, input_shape=(self.config.generator_noise_input_dim, ), activation='relu'))
        model.add(BatchNormalization(momentum=self.config.batchnormalization_momentum))
        model.add(Reshape((7, 7, 256)))

        model.add(Conv2DTranspose(128,kernel_size=3, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=self.config.batchnormalization_momentum))

        model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=self.config.batchnormalization_momentum))

        model.add(Conv2DTranspose(32, kernel_size=3,  padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=self.config.batchnormalization_momentum))

        model.add(Conv2DTranspose(self.config.discriminator_image_input_dim[2], kernel_size=3,
                                  padding='same', activation='tanh'))

        model.summary()

        noise = Input(shape=(self.config.generator_noise_input_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.config.condational_label_num, self.config.generator_noise_input_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)


    def train(self, train_data_gener, epoch, k, batch_size=256):
        """
        CGan训练函数
        :param train_data_gener:训练用数据生成器
        :param epoch: 周期数
        :param k: 每轮判别器训练次数
        :param batch_size: 小批量样本规模
        :return:
        """
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = os.path.join(self.config.model_dir, time)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        train_result_path = os.path.join(self.config.train_result_dir, time)
        if not os.path.exists(train_result_path):
            os.mkdir(train_result_path)

        cgan_losses = []
        d_losses = []
        for ep in range(1, epoch+1):
            #生成进度调
            length = train_data_gener.batch_num
            progbar = Progbar(length)
            print('Epoch {}/{}'.format(ep, epoch))
            iter = 0
            while True:
                # 遍历一次全部数据集，那么重新来结束while循环
                if train_data_gener.epoch != ep:
                    break

                #获取真实图片，并构造真图对应的标签
                batch_real_images, condation_labels,batch_noises = train_data_gener.next_batch()
                batch_size = len(batch_noises)
                real_images_score = np.ones((batch_size, 1))

                d_loss = []
                for i in np.arange(k):
                    #构造假图标签， 合并真图和假图对应的标签
                    fake_images_score = np.zeros((batch_size, 1))
                    batch_fake_images = self.generator_model.predict([batch_noises, condation_labels])
                    #训练判别器
                    real_d_loss = self.discriminator_model.train_on_batch(x=[batch_real_images, condation_labels],
                                                                          y=real_images_score)
                    fake_d_loss = self.discriminator_model.train_on_batch(x=[batch_fake_images, condation_labels],
                                                                          y=fake_images_score)
                    d_loss.append(list(0.5*np.add(real_d_loss, fake_d_loss)))

                d_losses.append(list(np.average(d_loss, axis=0)))

                #生成batch_size噪声训练生成器
                batch_num_labels = np.ones((batch_size, 1))
                batch_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
                cgan_loss = self.cgan.train_on_batch([batch_noises, batch_labels], y=batch_num_labels)   #使用的是同一个噪声输入
                cgan_losses.append(cgan_loss)

                #更新进度条
                progbar.update(iter, [('dcgan_loss', cgan_losses[iter]),
                                      ('discriminator_loss', d_losses[iter][0]),
                                      ('acc', d_losses[iter][1])])
                iter += 1

            #固定间隔存储model and result
            if ep % self.config.save_epoch_interval == 0:
                model_cgan = "Epoch{}dcgan_loss{}discriminator_loss{}.h5".format(ep, np.average(cgan_losses),
                                                                                 np.average(d_losses, axis=0)[0],
                                                                                 np.average(d_losses, axis=0)[1])
                self.cgan.save(os.path.join(model_path, model_cgan))
            if ep % self.config.savef_result_interval == 0:
                save_dir = os.path.join(train_result_path, "Epoch{}".format(ep))
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                self.save_image(int(ep), save_dir)


    def save_image(self, epoch, save_path):
        """
        这是保存生成图片的函数
        :param epoch:周期数
        :param save_path: 图片保存地址
        :return:
        """
        rows, cols = 10, 10

        fig, axs = plt.subplots(rows, cols)
        for i in range(rows):
            label = np.array([i] * rows).astype(np.int32).reshape(-1, 1)
            noise = np.random.normal(0, 1, (cols, 100))
            images = self.generator_model.predict([noise, label])
            images = 127.5 * images + 127.5
            cnt = 0
            for j in range(cols):
                # img_path = os.path.join(save_path, str(cnt) + ".png")
                # cv2.imwrite(img_path, images[cnt])
                # axs[i, j].imshow(image.astype(np.int32)[:,:,0])
                axs[i, j].imshow(images[cnt, :, :, 0].astype(np.int32), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(save_path, "mnist-{}.png".format(epoch)), dpi=600)
        plt.close()

    def generate_image(self, label):
        """
        这是伪造一张图片的函数
        :param label:标签
        """
        noise = truncnorm.rvs(-1, 1, size=(1, self.config.generator_noise_input_dim))
        label = np.array([label]).T
        image = self.generator_model.predict([noise, label])[0]
        image = 127.5 * (image + 1)
        return image


def make_trainable(net, val):
    """ Freeze or unfreeze layers
    """
    net.trainable = val
    for l in net.layers: l.trainable = val
