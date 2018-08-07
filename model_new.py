import tensorflow as tf
import numpy as np

import flag




class Classifier:
    def __init__(self):
        # Beta for L2 regularization
        beta = 0.01

        self.X = tf.placeholder(tf.float32, [None, flag.node_num, flag.total_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])
        self.learning_rate = 0.002
        self.dropout_rate = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        using_x = tf.reshape(self.X, [-1, flag.node_num, flag.total_dim, 1])

        """ image cnn
        conv1 = tf.layers.conv2d(using_x, filters=32, kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(pool1,filters=64,kernel_size=[5,5],padding="same",activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2,pool_size=[2,2],strides=2)

        self.hidden = tf.layers.dense(inputs=tf.reshape(pool2,[-1,25*7*64]),units=3000)
        self.hidden2 = tf.layers.dense(inputs=self.hidden,units=500)
        self.treat = tf.layers.dense(inputs=self.hidden2,units=30)
        """

        # sunno cnn
        conv1 = tf.layers.conv2d(using_x, filters=512, kernel_size=[4, flag.total_dim], activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[flag.node_num - 4 + 1, 1], strides=1)
        drop1 = tf.layers.dropout(pool1,self.dropout_rate)
        result1 = tf.reshape(drop1, (-1, 1, 512))
        self.r_result1 = tf.reshape(result1, (-1, 512))

        conv2 = tf.layers.conv2d(using_x, filters=512, kernel_size=[5, flag.total_dim], activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[flag.node_num - 5 + 1, 1], strides=1)
        drop2 = tf.layers.dropout(pool2, self.dropout_rate)
        result2 = tf.reshape(drop2, (-1, 1, 512))
        self.r_result2 = tf.reshape(result2, (-1, 512))

        conv3 = tf.layers.conv2d(using_x, filters=512, kernel_size=[6, flag.total_dim], activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[flag.node_num - 6 + 1, 1], strides=1)
        drop3 = tf.layers.dropout(pool3, self.dropout_rate)
        result3 = tf.reshape(drop3, (-1, 1, 512))
        self.r_result3 = tf.reshape(result3, (-1, 512))

        conv4 = tf.layers.conv2d(using_x, filters=256, kernel_size=[7, flag.total_dim], activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[flag.node_num - 7 + 1, 1], strides=1)
        drop4 = tf.layers.dropout(pool4, self.dropout_rate)
        result4 = tf.reshape(drop4, (-1, 1, 256))
        self.r_result4 = tf.reshape(result4, (-1, 256))

        conv5 = tf.layers.conv2d(using_x, filters=256, kernel_size=[8, flag.total_dim], activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[flag.node_num - 8 + 1, 1], strides=1)
        drop5 = tf.layers.dropout(pool5, self.dropout_rate)
        result5 = tf.reshape(drop5, (-1, 1, 256))
        self.r_result5 = tf.reshape(result5, (-1, 256))

        self.treat = tf.concat([self.r_result1, self.r_result2, self.r_result3,self.r_result4,self.r_result5], axis=1)
        self.treat = tf.layers.dropout(tf.layers.dense(inputs=self.treat,units=700,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)),self.dropout_rate)
        self.treat = tf.layers.dropout(tf.layers.dense(inputs=self.treat, units=200, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)),self.dropout_rate)
        self.treat = tf.layers.dropout(tf.layers.dense(inputs=self.treat, units=60, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)),self.dropout_rate)
        self.treat = tf.layers.dropout(tf.layers.dense(inputs=self.treat, units=20, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)),self.dropout_rate)

        # haesung deep
        # conv1 = tf.layers.conv2d(using_x, filters=512, kernel_size=[4, flag.node_dim], strides=1, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # drop1 = tf.layers.dropout(conv1, self.dropout_rate)
        # conv2 = tf.layers.conv2d(drop1, filters=512, kernel_size=[4, 1], strides=4, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # drop2 = tf.layers.dropout(conv2, self.dropout_rate)
        # conv3 = tf.layers.conv2d(drop2, filters=512, kernel_size=[4, 1], strides=1, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # drop3 = tf.layers.dropout(conv3, self.dropout_rate)
        # conv4 = tf.layers.conv2d(drop3, filters=512, kernel_size=[4, 1], strides=4, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # drop4 = tf.layers.dropout(conv4, self.dropout_rate)
        # conv5 = tf.layers.conv2d(drop4, filters=512, kernel_size=[4, 1], strides=1, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # drop5 = tf.layers.dropout(conv5, self.dropout_rate)
        # conv6 = tf.layers.conv2d(drop5, filters=512, kernel_size=[4, 1], strides=4, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # drop6 = tf.layers.dropout(conv6, self.dropout_rate)
        # conv7 = tf.layers.conv2d(drop6, filters=512, kernel_size=[4, 1], strides=1, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # drop7 = tf.layers.dropout(conv7, self.dropout_rate)
        # conv8 = tf.layers.conv2d(drop7, filters=512, kernel_size=[4, 1], strides=4, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # drop8 = tf.layers.dropout(conv8, self.dropout_rate)
        #
        # conv9 = tf.layers.conv2d(drop8, filters=512, kernel_size=[4, 1], strides=1, padding="SAME",
        #                          activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # drop8 = tf.layers.dropout(conv9, self.dropout_rate)
        #
        # shape_list = drop8.get_shape().as_list()
        # shape_add = shape_list[1] * shape_list[2] * shape_list[3]
        # conv_flat = tf.reshape(conv9, (-1, shape_add))
        # hidden = tf.layers.dense(inputs=conv_flat, units=150, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # drop_hidden = tf.layers.dropout(hidden, self.dropout_rate)
        # self.hidden2 = tf.layers.dense(inputs=drop_hidden, units=50, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # drop_hidden2 = tf.layers.dropout(self.hidden2, self.dropout_rate)
        # self.treat = tf.layers.dropout(tf.layers.dense(inputs=drop_hidden2, units=10, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)), self.dropout_rate)
        #
        self.logits = tf.layers.dense(inputs=self.treat, units=1)
        self.hypothesis = self.logits
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y))
        #l2_loss = tf.losses.get_regularization_loss()
        #self.cost += l2_loss
        self.cost_summ = tf.summary.scalar("cost", self.cost)
        # self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y*tf.log(self.hypothesis)))
        self.predict = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.Y), dtype=tf.float32))
        self.accuracy_summ = tf.summary.scalar("accuracy", self.accuracy)

        # tvars = tf.trainable_variables() grads, global_norm = tf.clip_by_global_norm(tf.gradients(self.cost,
        # tvars), 5.0) self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(grads,tvars),
        # global_step=self.global_step)
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.cost,
            global_step=self.global_step,
            learning_rate=self.learning_rate-tf.multiply(0.00000022,tf.cast(self.global_step,tf.float32)),
            optimizer='Adam',
        )
#        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost,global_step=self.global_step)

