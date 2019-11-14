import tensorflow as tf
import numpy as np
import tensorflow.nn as nn

from sklearn.metrics import f1_score


flag = True

def top_k_features(adj_m, fea_m, k, scope):
    adj_m = tf.expand_dims(adj_m, axis=1, name=scope+'/expand1')
    fea_m = tf.expand_dims(fea_m, axis=-1, name=scope+'/expand2')
    feas = tf.multiply(adj_m, fea_m, name=scope+'/mul')
    feas = tf.transpose(feas, perm=[2, 1, 0], name=scope+'/trans1')
    top_k = tf.nn.top_k(feas, k=k, name=scope+'/top_k').values
    #pre, post = tf.split(top_k, 2, axis=2, name=scope+'/split')

    top_k = tf.concat([fea_m, top_k], axis=2, name=scope+'/concat')
    top_k = tf.transpose(top_k, perm=[0, 2, 1], name=scope+'/trans2')

    return top_k

def top_k_features_no_concat(adj_m, fea_m, k, scope):

    adj_m = tf.expand_dims(adj_m, axis=1, name=scope+'/expand1')
    fea_m = tf.expand_dims(fea_m, axis=-1, name=scope+'/expand2')
    feas = tf.multiply(adj_m, fea_m, name=scope+'/mul')
    feas = tf.transpose(feas, perm=[2, 1, 0], name=scope+'/trans1')
    top_k = tf.nn.top_k(feas, k=k, name=scope+'/top_k').values
    #pre, post = tf.split(top_k, 2, axis=2, name=scope+'/split')
    top_k = tf.transpose(top_k, perm=[0, 2, 1], name=scope+'/trans2')
    fea_m = tf.transpose(fea_m, perm=[0, 2, 1], name=scope+'/trans3')

    return fea_m, top_k

def top_k_features_concat(adj_m, fea_m, k, count_adj, scope):
    adj_m = tf.expand_dims(adj_m, axis=1, name=scope+'/expand1')
    fea_m = tf.expand_dims(fea_m, axis=-1, name=scope+'/expand2')
    feas = tf.multiply(adj_m, fea_m, name=scope+'/mul')
    feas = tf.transpose(feas, perm=[2, 1, 0], name=scope+'/trans1')
    print(feas, 'dfdfdfdfddf')
    top_k = tf.nn.top_k(feas, k=k, name=scope+'/top_k').values
    #pre, post = tf.split(top_k, 2, axis=2, name=scope+'/split')
    print(top_k.shape, 'top_k')
    print(fea_m.shape, 'top_k')
    top_k = tf.concat([fea_m, top_k], axis=2, name=scope+'/concat')
    top_k = tf.transpose(top_k, perm=[0, 2, 1], name=scope+'/trans2')
    print(top_k.shape, 'top_k')
    return top_k


def mylayer(center, top_k, coord_tensor, kp, is_training):

        mb = top_k.shape[0]
        n_channels = top_k.shape[2]
        dd = top_k.shape[1]
        x_flat = tf.concat([top_k, coord_tensor], 2)
        x_i = tf.expand_dims(x_flat, 1)  
        x_i = tf.tile(input=x_i, multiples=[1, 12, 1, 1])  
        x_j = tf.expand_dims(x_flat, 2)  
        x_j = tf.tile(input=x_j, multiples=[1, 1, 12, 1])  

        # concatenate all together
        x_full = tf.concat([x_i, x_j], 3)  

        # reshape for passing through network
        x_ = tf.reshape(x_full, [mb*dd*dd, 100])

        x_ = tf.layers.dense(x_, 128)
        x_ = nn.relu(x_)
        x_ = tf.layers.dense(x_, 128)
        x_ = nn.relu(x_)
        x_ = tf.layers.dense(x_, 128)
        x_ = nn.relu(x_)
        x_ = tf.layers.dense(x_, 128)
        x_ = nn.relu(x_)
        # reshape again and sum
        x_g = tf.reshape(x_, [mb, dd*dd, 128])
        x_g = tf.reduce_sum(x_g, 1)
        x_g = tf.squeeze(x_g)

        """f"""
        x_f = tf.layers.dense(x_g, 128)
        x_f = nn.relu(x_f)
        x = tf.layers.dense(x_f, 128)
        x = nn.relu(x)
        x = dropout(x, kp, is_train=is_training, scope='sss')
        # x = conv1d(x,32, 1, scope='sss')
        x = tf.layers.dense(x, 12)
        # x = nn.relu6(x)  
        # x = nn.softmax(x) 
        # x = batch_norm(x, is_train=is_training, scope='sss'+'/norm2', act_fn= tf.nn.relu)
        # x = tf.nn.elu(x)  
        # x = tf.sigmoid(x)
        # x = tf.log_sigmoid(x)
        x = nn.relu(x)  
        # x = tf.nn.log_softmax(x)
        # x = nn.sigmoid(x) 

        return x


def myconcat(mytensor, top_k, center):
        
        mytensor = tf.expand_dims(mytensor, axis=2)
        mytensor = tf.tile(input=mytensor, multiples=[1,1,48])
        temp = tf.multiply(mytensor, top_k)
        # center = tf.reduce_sum(center, axis=1)
        outs = tf.concat([center, temp], axis=1)
        # outs = outs.sum(axis=1)
        # outs = tf.reduce_max(outs, 1)
        outs = tf.reduce_sum(outs, 1)
        outs = nn.sigmoid(outs)
        return outs


def simple_conv(adj_m, outs, num_out, adj_keep_r, keep_r, is_train, scope,
                act_fn=tf.nn.elu, norm=True, **kw):
    adj_m = dropout(adj_m, adj_keep_r, is_train, scope+'/drop1')
    outs = dropout(outs, keep_r, is_train, scope+'/drop2')
    outs = fully_connected(outs, num_out, scope+'/fully', None)
    
    outs = tf.matmul(adj_m, outs, name=scope+'/matmul')

    #if norm:
    #    outs = batch_norm(outs, is_train, scope=scope+'/norm', act_fn=None)
    outs = outs if not act_fn else act_fn(outs, scope+'/act')
    print(act_fn, 'act_fn')
    return outs


def graph_conv(adj_m, outs, num_out, adj_keep_r, keep_r, is_train, scope, k=5,
               coord_tensor=None, act_fn=tf.nn.relu6,count_adj=None, **kw):
    num_in = outs.shape[-1].value
    adj_m = dropout(adj_m, adj_keep_r, is_train, scope+'/drop1')
    center, top_k = top_k_features_no_concat(adj_m, outs, k, scope+'/top_k')
    outs = top_k_features(adj_m, outs, k,  scope+'/top_kkk')
    if top_k.shape[2] == 48:
        mytensor = mylayer(center, top_k, coord_tensor, 0.4, is_train)
        outs = myconcat(mytensor, top_k, center)
    else:
        outs = tf.reduce_sum(outs, axis=1)
        outs = nn.sigmoid(outs)
        
        # outs = nn.dropout(outs, keep_prob=0.5)
        # outs = tf.layers.dense(outs, 32)
        # outs = batch_norm(outs, is_train=is_train, scope='sdfdfss'+'/norm2', act_fn= tf.nn.relu)
        # outs = nn.elu(outs)
        # outs = 
        # return outs
        '''
        outs = dropout(outs, keep_r, is_train, scope+'/drop1')
        outs = conv1d(outs, 20, (k+1)//2+1, scope+'/conv1', None, True)
        outs = act_fn(outs, scope+'act1') if act_fn else outs
        outs = dropout(outs, keep_r, is_train, scope+'/drop2')
        outs = conv1d(outs, 32, k//2+1, scope+'/conv2', None)
        outs = tf.squeeze(outs, axis=[1], name=scope+'/squeeze')
        outs = act_fn(outs, scope+'act1') if act_fn else outs
        outs = batch_norm(outs, True, scope+'/norm2', act_fn)
        print(outs, 'xianyan')
        '''

    return outs


def fully_connected(outs, dim, scope, act_fn=tf.nn.softmax):
    outs = tf.contrib.layers.fully_connected(
        outs, dim, activation_fn=None, scope=scope+'/dense',
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=tf.contrib.layers.xavier_initializer())
        #weights_initializer=tf.random_normal_initializer(),
        #biases_initializer=tf.random_normal_initializer())
    return act_fn(outs, scope+'/act') if act_fn else outs


def conv1d(outs, num_out, k, scope, act_fn=tf.nn.relu, use_bias=False):
    l2_func = tf.contrib.layers.l2_regularizer(5e-4, scope)
    outs = tf.layers.conv1d(
        outs, num_out, k, activation=act_fn, name=scope+'/conv',
        padding='valid', use_bias=use_bias,
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    return outs


def chan_conv(adj_m, outs, num_out, keep_r, is_train, scope,
              act_fn=tf.nn.relu):
    outs = dropout(outs, keep_r, is_train, scope)
    outs = tf.matmul(adj_m, outs, name=scope+'/matmul')
    in_length = outs.shape.as_list()[-1]
    outs = tf.expand_dims(outs, axis=-1, name=scope+'/expand')
    kernel = in_length - num_out + 1
    outs = conv1d(outs, 1, kernel, scope+'/conv', act_fn)
    outs = tf.squeeze(outs, axis=[-1], name=scope+'/squeeze')
    return batch_norm(outs, True, scope, act_fn)


def dropout(outs, keep_r, is_train, scope):
    if keep_r < 1.0:
        return tf.contrib.layers.dropout(
            outs, keep_r, is_training=is_train, scope=scope)
    return outs


def batch_norm(outs, is_train, scope, act_fn=tf.nn.relu):
    return tf.contrib.layers.batch_norm(
        outs, scale=True,
        activation_fn=act_fn, fused=True,
        is_training=is_train, scope=scope,
        updates_collections=None)


def masked_softmax_cross_entropy(preds, labels, mask, name='loss'):
    with tf.variable_scope(name):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask, name='accuracy'):

    with tf.variable_scope(name):
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
      
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)


def masked_accuracy_batch(preds, labels, mask, name='accuracy'):
    with tf.variable_scope(name):
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        # print(correct_prediction)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        # print(mask)
        mask = tf.cast(mask, dtype=tf.float32)
        # mask /= tf.reduce_mean(mask)
        # accuracy_all *= mask
        accuracy = tf.multiply(correct_prediction, mask)
        accuracy_all = tf.cast(accuracy, tf.float32)
        # count = tf.sum(accuracy_all)
        count = tf.reduce_sum(accuracy_all)
        count_mask = tf.reduce_sum(mask)
        return count, count_mask

def score(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.round().astype(np.int32)
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(f1_score(y_true[:,i], y_pred[:,i], average="micro"))
    #return max(scores)
    return sum(scores) / len(scores)
    #return scores/y_true.shape[1]
