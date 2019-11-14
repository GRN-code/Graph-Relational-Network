import numpy as np
import tensorflow.nn as nn
import tensorflow as tf
import pysnooper



class RN(object):



    def __init__(self):
        self.batch_size = 605

        self.coord_oi = tf.Variable(tf.zeros((self.batch_size, 2)))
        self.coord_oj = tf.Variable(tf.zeros((self.batch_size, 2)))

        def cvt_coord(i):
            return [(i / 2 - 2) / 2., (i % 2 - 2) / 2.]

        self.coord_tensor = tf.Variable(tf.zeros((self.batch_size, 38, 2)))
        # if args.cuda:
        #     self.coord_tensor = self.coord_tensor.cuda()
        # self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((self.batch_size, 38, 2))
        for i in range(38):
            np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
        self.coord_tensor = tf.convert_to_tensor(np_coord_tensor, tf.float32)
        def init_weights(shape):
            return tf.Variable(tf.random_normal(shape, stddev=0.01))

        self.w = init_weights([3, 3, 8, 8])
        self.w1 = init_weights([3, 3, 8, 8])
        self.w2 = init_weights([3, 3, 8, 8])
        self.w3 = init_weights([3, 3, 8, 8])
        self.w4 = init_weights([3, 3, 8, 8])
        self.w5 = init_weights([3, 3, 8, 8])
        self.w6 = init_weights([3, 3, 8, 8])
        # print(self.coord_tensor, 'dddd')
        # exit()

    def train(self, input, h, ffd_drop, hid_units, adj, number):
        # self.input = tf.placeholder(dtype=tf.float64, shape=(self.batch_size, 3025, 1870))
        # self.h = tf.placeholder(dtype=tf.float64, shape=(self.batch_size, 64))


        embed_list = []
        """g"""
        input = tf.expand_dims(input, 0)
        # print(input.shape, 'fdfdfdfdfdfd')
        # exit()
        if ffd_drop != 0.0:
            seq = tf.nn.dropout(input, 1.0 - ffd_drop)
            input = tf.layers.conv1d(seq, hid_units, 1, use_bias=False)
        # print(input.shape, 'kokokoko')
        # print(h.shape)
        # print(ffd_drop)
        # print(hid_units)
        # print(adj)
        # exit()
        # adj = adj.astype(np.float32)
        # input1 = tf.matmul(adj, input) + input
        # print(input_1.shape)
        # exit()
        print(number)
        print(self.batch_size)

        for i in range(self.batch_size):
            one_adj = adj[0][i + number*self.batch_size].reshape(adj[0].shape[0], 1)
            input1 = one_adj * input
            # embed_list.append(input1)
            # input = tf.concat(embed_list, axis=1)
            # print(multi_embed.shape, 'lplplplpllp')
            # print(input1.shape)
            # exit()

            # input1 = tf.reshape(input1, [55, 55, 8])
            # print(input1.shape)
            # exit()
            # input1 = tf.tile(input=input1, multiples=[3025, 1, 1])
            # # self.input = input
            # # self.h = h

            # input1 = tf.nn.conv2d(input1, self.w3, strides=[1, 3, 3, 1], padding='SAME')
            # input1 = tf.nn.conv2d(input1, self.w4, strides=[1, 3, 3, 1], padding='SAME')
            # input1 = tf.nn.conv2d(input1, self.w5, strides=[1, 3, 3, 1], padding='SAME')
            # input1 = tf.nn.conv2d(input1, self.w6, strides=[1, 3, 3, 1], padding='SAME')
            # # print(input.shape, 'ffff')
            # # exit()
            # # input1 = tf.squeeze(input1)
            # # print(input1.shape)
            input1 = tf.reshape(input1, [1, -1, 8])
            # print(input1.shape)

            embed_list.append(input1)
        input = tf.concat(embed_list, axis=0)
        print(input.shape)
        input1 = tf.expand_dims(input, 0)
        #
        input1 = tf.nn.conv2d(input1, self.w, strides=[1, 1, 3, 1], padding='SAME')
        input1 = tf.nn.conv2d(input1, self.w1, strides=[1, 1, 3, 1], padding='SAME')
        input1 = tf.nn.conv2d(input1, self.w2, strides=[1, 1, 3, 1], padding='SAME')
        input1 = tf.nn.conv2d(input1, self.w3, strides=[1, 1, 3, 1], padding='SAME')
        # print(input1.shape)
        # exit()
        input = tf.squeeze(input1)
        h = h[number*self.batch_size : self.batch_size + number*self.batch_size]
        # print(input.shape, 'ffff')
        # exit()
        # input = tf.reshape(input, [3025, -1, 8])
        # input = tf.expand_dims(input, 0)
        # input = tf.tile(input=input, multiples=[self.batch_size, 1, 1])
        mb = input.shape[0]
        # print(mb, 'ssssss')
        n_channels = input.shape[2]
        dd = input.shape[1]
        # print(input.shape, 'jijijjj')
        # exit()
        # print(mb, n_channels, dd, "opopopopooopopoop")
        # print(input.shape,'fffffff')
        x_flat = tf.concat([input, self.coord_tensor], 2)
        # print(h, 'qqqqq')
        # h = tf.squeeze(h)
        # self.h = tf.expand_dims(self.h, 0)
        print(h.shape, 'jojojoj')
        qst = tf.expand_dims(h, 1)
        qst = tf.tile(input=qst, multiples=[1, 38, 1])
        qst = tf.expand_dims(qst, 2)

        x_i = tf.expand_dims(x_flat, 1)  # (64x1x25x26+5)
        x_i = tf.tile(input=x_i, multiples=[1, 38, 1, 1])  # (64x25x25x26+5)
        x_j = tf.expand_dims(x_flat, 2)  # (64x25x1x26+5)
        x_j = tf.concat([x_j, qst], 3)
        x_j = tf.tile(input=x_j, multiples=[1, 1, 38, 1])  # (64x25x25x26+5)

        # concatenate all together
        x_full = tf.concat([x_i, x_j], 3)  # (64x25x25x2*26+5)
        # print(x_f)
        # print(x_full.shape, 'uiuiuiuiu')
        # exit()
        # reshape for passing through network
        x_ = tf.reshape(x_full, [mb*dd*dd, 84])
        # print(x_.shape, 'kokokkoko')
        # exit()

        x_ = tf.layers.dense(x_, 128)
        x_ = nn.relu(x_)
        x_ = tf.layers.dense(x_, 128)
        x_ = nn.relu(x_)
        x_ = tf.layers.dense(x_, 128)
        x_ = nn.relu(x_)
        x_ = tf.layers.dense(x_, 128)
        x_ = nn.relu(x_)

        # reshape again and sum
        # print(x_.shape, 'jijijijijiji')
        # exit()
        # print(mb, 'fdfdfdfdfffd')
        # print(dd*dd, 'jijijijijj')
        x_g = tf.reshape(x_, [mb, dd*dd, 128])
        # print(x_g.shape, 'llolooollololo')
        # exit()
        # print(x_g.shape, 'jiijijjji')
        x_g = tf.reduce_sum(x_g, 1)
        # print(x_g.shape, 'kokokokok')
        # exit()
        # print(x_g.shape, 'huuhuhuhu')
        x_g = tf.squeeze(x_g)
        # print(x_g.shape, 'huuhuhuhu')
        # exit()
        """f"""
        x_f = tf.layers.dense(x_g, 128)
        x_f = nn.relu(x_f)
        x = tf.layers.dense(x_f, 128)
        x = nn.relu(x)
        x = nn.dropout(x, keep_prob=0.5)
        x = tf.layers.dense(x, 64)
        return x




