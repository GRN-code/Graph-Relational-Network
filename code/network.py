import os
import time
import random
import numpy as np
import networkx as nx
import tensorflow as tf
import ops
import time
from utils import load_data, preprocess_features, preprocess_adj
from batch_utils import get_sampled_index, get_indice_graph


class GraphNet(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        self.process_data()
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')
        self.test_summary = self.config_summary('test')

    def inference(self, outs):

        outs = getattr(ops, self.conf.first_conv)(
            self.normed_matrix, outs, 4*self.conf.ch_num, self.conf.adj_keep_r,
            self.conf.keep_r, self.is_train, 'conv_s', act_fn=None)
        for layer_index in range(self.conf.layer_num):
            cur_outs= getattr(ops, self.conf.second_conv)(
                self.normed_matrix, outs, self.conf.ch_num, self.conf.adj_keep_r,
                self.conf.keep_r, self.is_train, 'conv_%s' % (layer_index+1),
                coord_tensor=self.coord_tensor, act_fn=None, k=self.conf.k, count_adj=layer_index)
            outs = tf.concat([outs, cur_outs], axis=1, name='concat_%s' % layer_index)
        outs = ops.simple_conv(
            self.normed_matrix, outs, self.conf.class_num, self.conf.adj_keep_r,
            self.conf.keep_r, self.is_train, 'conv_f', act_fn=None, norm=False)
        # exit()
        return outs

    def get_optimizer(self, lr):
        # return tf.contrib.opt.NadamOptimizer(lr)
        # return tf.train.GradientDescentOptimizer(lr)
        # return tf.train.AdamOptimizer(lr)
        # return tf.contrib.opt.AdamWOptimizer(lr)
        return tf.train.AdamOptimizer(lr)

    def process_data(self):
        data = load_data('cora')
        adj, feas = data[:2]
        self.adj = adj.todense()
        self.normed_adj = preprocess_adj(adj)
        self.feas = preprocess_features(feas, False)
        self.y_train, self.y_val, self.y_test = data[2:5]
        self.train_mask, self.val_mask, self.test_mask = data[5:]


        print((self.train_mask==True).sum())

        print((self.val_mask==True).sum())
        print((self.test_mask==True).sum())

    def configure_networks(self):
        self.build_network()
        self.cal_loss()
        optimizer = self.get_optimizer(self.conf.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        self.seed = int(time.time())
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        if self.conf.is_train:
            self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.print_params_num()

    def build_network(self):
        self.labels_mask = tf.placeholder(tf.int32, self.conf.batch_size, name='labels_mask')
        self.matrix = tf.placeholder(tf.int32, [self.conf.batch_size, self.conf.batch_size], name='matrix')
        self.normed_matrix = tf.placeholder(tf.float32, [self.conf.batch_size, self.conf.batch_size], name='normed_matrix')
        self.inputs = tf.placeholder(tf.float32, [self.conf.batch_size, self.feas.shape[1]], name='inputs')
        self.labels = tf.placeholder(tf.int32, [self.conf.batch_size, self.conf.class_num], name='labels')
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.count_adj1 = tf.placeholder(tf.float32, [self.conf.batch_size, self.conf.batch_size], name='count_adj')
        def cvt_coord(i):
            return [ 0., (i % 12 - 2) / 2.]
        np_coord_tensor = np.ones((self.conf.batch_size, 12, 2))
        for i in range(12):
            np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
        coord_tensor_1 = tf.convert_to_tensor(np_coord_tensor, tf.float32)
        self.coord_tensor = tf.Variable(coord_tensor_1)
        self.preds = self.inference(self.inputs)



    def cal_loss(self):
        with tf.variable_scope('loss'):
            self.class_loss = ops.masked_softmax_cross_entropy(
                self.preds, self.labels, self.labels_mask)
            self.regu_loss = 0
            # for var in tf.trainable_variables():
            #     if var.name in ['bias', 'gamma', 'b', 'g', 'beta']:
            #         continue
            #     self.regu_loss += self.conf.weight_decay * tf.nn.l2_loss(var)
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                        in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.conf.weight_decay

            self.loss_op = self.class_loss + lossL2
            self.accuracy_op = ops.masked_accuracy(self.preds, self.labels, self.labels_mask)

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/class_loss', self.class_loss))
        if name == 'train':
            summarys.append(tf.summary.scalar(name+'/regu_loss', self.regu_loss))
        summary = tf.summary.merge(summarys)
        return summary

    def save_summary(self, summary, step):
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        self.transductive_train()

    def transductive_train(self):
        feed_train_dict = self.pack_trans_dict('train')
        feed_valid_dict = self.pack_trans_dict('valid')
        feed_test_dict = self.pack_trans_dict('test')
        stats = [0, 0, 0]
        best_acc = 0

        for epoch_num in range(self.conf.max_step+1):
            first = time.time()
            train_loss, _, summary, train_accuracy = self.sess.run(
                [self.loss_op, self.train_op, self.train_summary, self.accuracy_op],
                feed_dict=feed_train_dict)
            second = time.time()
            # print(second-first)
            self.save_summary(summary, epoch_num+self.conf.reload_step)
            summary, valid_accuracy = self.sess.run(
                [self.valid_summary, self.accuracy_op],
                feed_dict=feed_valid_dict)
            first = time.time()
            self.save_summary(summary, epoch_num+self.conf.reload_step)
            summary, test_accuracy = self.sess.run(
                [self.test_summary, self.accuracy_op],
                feed_dict=feed_test_dict)
            second = time.time()
            # print(second-first)
            if test_accuracy > best_acc:
                print(test_accuracy, 'best')
                best_acc = test_accuracy
            self.save_summary(summary, epoch_num+self.conf.reload_step)
            if valid_accuracy >= stats[0]:
                stats[0], stats[1], stats[2] = valid_accuracy, 0, max(test_accuracy, stats[2])
            else:
                stats[1] += 1
            if epoch_num and epoch_num % 100 == 0:
                self.save(epoch_num)
            print('step: %d --- loss: %.4f, train: %.3f, val: %.3f' %(
                epoch_num, train_loss, train_accuracy, valid_accuracy))
            print('Test accuracy -----> ', self.seed, best_acc)
            
            
            if stats[1] > 150 and epoch_num > 150:
                print('Test accuracy -----> ', self.seed, best_acc)
                file = open('1.txt', 'a+')
                file.write(str(best_acc)+'\n')
                break

    def pack_trans_dict(self, action):
        feed_dict = {
            self.matrix: self.adj, self.normed_matrix: self.normed_adj,
            self.inputs: self.feas}
        if action == 'train':  #644
            feed_dict.update({
                self.labels: self.y_train, self.labels_mask: self.train_mask,
                self.is_train: True})
            if self.conf.use_batch:
                indices = get_indice_graph(
                    self.adj, self.train_mask, self.conf.batch_size, 1.0)
                new_adj = self.adj[indices,:][:,indices]
                new_normed_adj = self.normed_adj[indices,:][:,indices]

                feed_dict.update({
                    self.labels: self.y_train[indices],
                    self.labels_mask: self.train_mask[indices],
                    self.matrix: new_adj, self.normed_matrix: new_normed_adj,
                    self.inputs: self.feas[indices]})
                print(len(indices), 'train')

        elif action == 'valid':  #1484
            feed_dict.update({
                self.labels: self.y_val, self.labels_mask: self.val_mask,
                self.is_train: False})
            if self.conf.use_batch:
                indices = get_indice_graph(
                    self.adj, self.val_mask, self.conf.batch_size, 1.0)
                new_adj = self.adj[indices,:][:,indices]
                new_normed_adj = self.normed_adj[indices,:][:,indices]
                feed_dict.update({
                    self.labels: self.y_val[indices],
                    self.labels_mask: self.val_mask[indices],
                    self.matrix: new_adj, self.normed_matrix: new_normed_adj,
                    self.inputs: self.feas[indices]})
                print(len(indices), 'val')
                
        else:  #2190
            feed_dict.update({
                self.labels: self.y_test, self.labels_mask: self.test_mask,
                self.is_train: False})
            if self.conf.use_batch:
                indices = get_indice_graph(
                    self.adj, self.test_mask, self.conf.batch_size, 1.0)
                new_adj = self.adj[indices,:][:,indices]
                new_normed_adj = self.normed_adj[indices,:][:,indices]
                feed_dict.update({
                    self.labels: self.y_test[indices],
                    self.labels_mask: self.test_mask[indices],
                    self.matrix: new_adj, self.normed_matrix: new_normed_adj,
                    self.inputs: self.feas[indices]})
                print(len(indices), 'test')
        return feed_dict

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)

    def print_params_num(self):
        total_params = 0
        for var in tf.trainable_variables():
            print(var)
            total_params += var.shape.num_elements()
        print("The total number of params --------->", total_params)
