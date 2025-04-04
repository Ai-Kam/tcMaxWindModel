import tensorflow as tf
from tensorflow.keras import layers, models, utils
import numpy as np
import netCDF4 as nc4
import math

class nc4PyDataset(utils.PyDataset):

    def __init__(self, tp_list, stride=1, sample_length=4, traget_distance=4, batch_size=500, range=(0,1), max_wind_channnel=4, use_meta_channnel=None, **kwargs):
        super().__init__(**kwargs)
        len_tp = len(tp_list)
        self.tp_list = tp_list[int(len_tp*range[0]):int(len_tp*range[1])]
        print(len(self.tp_list))
        self.stride = stride
        self.sample_length = sample_length
        self.total_length = 0
        self.target_distance = traget_distance
        for tp in self.tp_list:
            nc = nc4.Dataset(tp, 'r')
            data_length = nc.dimensions['time'].size
            self.total_length += max(0, (data_length - self.sample_length - self.target_distance) // self.stride)
            nc.close()
        print(self.total_length)
        self.batch_size = batch_size
        self.current_tp = 0
        self.current_idt = self.sample_length
        self.max_wind_channnel = max_wind_channnel
        self.use_meta_channnel = use_meta_channnel
        self.field_shape = None
        self.meta_width = None

    def __len__(self):
        # Return number of batches.
        return math.ceil(self.total_length / self.batch_size)

    def __get1sample__(self):
        if self.current_tp >= len(self.tp_list):
            raise StopIteration

        nc = nc4.Dataset(self.tp_list[self.current_tp], 'r')
        nclen = nc.dimensions['time'].size
        while self.current_idt + self.target_distance >= nclen:
            nc.close()
            self.current_tp += 1
            self.current_idt = self.sample_length
            if self.current_tp >= len(self.tp_list):
                self.current_tp = 0
                self.current_idt = self.sample_length
            nc = nc4.Dataset(self.tp_list[self.current_tp], 'r')
            nclen = nc.dimensions['time'].size
        meta = nc['meta'][self.current_idt-self.sample_length:self.current_idt] if self.use_meta_channnel is None else nc['meta'][self.current_idt-self.sample_length:self.current_idt, self.use_meta_channnel]
        fields = nc['fields'][self.current_idt:self.current_idt+self.sample_length]
        target = nc['meta'][self.current_idt+self.target_distance, self.max_wind_channnel]
        nc.close()
        self.current_idt += self.stride

        return meta, fields, target
    
    def reset(self):
        self.current_tp = 0
        self.current_idt = self.sample_length

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, self.total_length)

        item = []
        for _ in range(high - low):
            item_temp = self.__get1sample__()
            item.append(item_temp)

        item1 = np.array([i[0] for i in item])
        item2 = np.array([i[1] for i in item])
        item3 = np.array([i[2] for i in item])

        return (item1, item2), item3

class MaxWindModel(models.Model):
    def __init__(self, horizontal_stride=4, kernel_size=4, **kwargs):
        trainable = kwargs.pop('trainable', True)
        super(MaxWindModel, self).__init__(trainable=trainable)
        self.horizontal_stride = (horizontal_stride, horizontal_stride)
        self.kernel_size = (kernel_size, kernel_size)
        self.transpose = layers.Permute((1, 3, 4, 2))
        self.conv1 = layers.Conv2D(32, self.kernel_size, activation='relu', strides=self.horizontal_stride)    # 16 x 16
        self.pool1 = layers.MaxPooling2D((2, 2))   # 8 x 8
        self.conv2 = layers.Conv2D(64, self.kernel_size, activation='relu', strides=self.horizontal_stride)    # 2 x 2
        self.pool2 = layers.MaxPooling2D((2, 2))   # 1 x 1
        self.concat = layers.Concatenate(axis=-1)
        self.lstml1 = layers.LSTM(128, activation='relu', return_sequences=True)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(16, activation='relu')
        self.dense3 = layers.Dense(1, activation='linear')

    def call(self, x_given, training=False):
        meta, fields = x_given
        x = self.transpose(fields)
        xs = tf.unstack(x, axis=1)
        for i in range(4):
            xs[i] = self.conv1(xs[i])
            xs[i] = self.pool1(xs[i])
            xs[i] = self.conv2(xs[i])
            xs[i] = self.pool2(xs[i])
        x = tf.stack(xs, axis=1)
        x = x[:,:,0,0,:]
        x = self.concat([meta, x])
        x = self.lstml1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def get_config(self):
        config = super(MaxWindModel, self).get_config()
        config.update({
            "horizontal_stride": self.horizontal_stride,
            "kernel_size": self.kernel_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class MaxWindModel2(models.Model):
    def __init__(self):
        super(MaxWindModel2, self).__init__()
        self.transpose = layers.Permute((1, 3, 4, 2))
        self.conv1 = layers.Conv2D(filters=16, kernel_size=(4, 4), padding='same', activation='relu', strides=(2, 2))    # 32 x 32
        self.conv2 = layers.Conv2D(filters=32, kernel_size=(4, 4), padding='same', activation='relu', strides=(2, 2))    # 16 x 16
        self.conv3 = layers.Conv2D(filters=64, kernel_size=(4, 4), padding='same', activation='relu', strides=(2, 2))    # 8 x 8
        self.conv4 = layers.Conv2D(filters=128, kernel_size=(4, 4), padding='same', activation='relu', strides=(2, 2))    # 4 x 4
        self.conv5 = layers.Conv2D(filters=256, kernel_size=(4, 4), activation='relu')    # 1 x 1
        self.concat = layers.Concatenate(axis=-1)
        self.lstml1 = layers.LSTM(256, activation='relu', return_sequences=True)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(16, activation='relu')
        self.dense3 = layers.Dense(1, activation='linear')


    def call(self, x_given, training=False):
        meta, fields = x_given
        x = self.transpose(fields)
        xs = tf.unstack(x, axis=1)
        for i in range(4):
            xs[i] = self.conv1(xs[i])
            xs[i] = self.conv2(xs[i])
            xs[i] = self.conv3(xs[i])
            xs[i] = self.conv4(xs[i])
            xs[i] = self.conv5(xs[i])
        x = tf.stack(xs, axis=1)
        print(x.shape)
        x = x[:,:,0,0,:]
        x = self.concat([meta, x])
        x = self.lstml1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
