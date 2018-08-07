import tensorflow as tf
import numpy as np
import time

import WordVecModeler_new
import flag
import utils_new
import model_new

if __name__ == '__main__':
    f_manager = utils_new.FileManager(flag.data_dir)
    root_nodes = f_manager.root_node_list
    wordVecModel = WordVecModeler_new.WordVecModeler(flag.node_dim,flag.word_vec_dim)
    wordVecModel.load_word_vec("word_model_50","GoogleNews-vectors-negative300.bin")
    answer_list = list()
    data_list = list()
    for idx, node in enumerate(root_nodes):
        tmp = utils_new.convert_node_to_vector_list(node, option=3,split_add=True)
        changeList = list()
        for element in tmp:
            node_vec = None
            word_vec = None
            if isinstance(element,np.ndarray):
                changeList.append(np.zeros(flag.total_dim))
                continue
            else:
                node_part = element.split("@")[0]
                if len(element.split("@")) ==1 :
                    word_list = list()
                else:
                    word_part = element.split("@")[1]
                    word_list = word_part.split(",")
            if wordVecModel.get_vector_from_node(node_part) is not None:
                node_vec = wordVecModel.get_vector_from_node(node_part)
            elif wordVecModel.get_vector_from_word(node_part) is None:
                node_vec = np.zeros(flag.node_dim)

            for one_word in word_list:
                if wordVecModel.get_vector_from_word(one_word) is not None:
                    c_vec = wordVecModel.get_vector_from_word(one_word)
                else:
                    c_vec = np.zeros(flag.word_vec_dim)
                if word_vec is None:
                    word_vec = c_vec
                else:
                    word_vec = np.concatenate((word_vec,c_vec))
            if word_vec is None:
                word_vec = list()
            if len(word_vec) < flag.word_vec_dim * flag.word_vec_num:
                zeros = np.zeros(flag.word_vec_dim * flag.word_vec_num - len(word_vec))
                word_vec = np.concatenate((word_vec,zeros))
            word_vec = word_vec[:flag.word_vec_dim * flag.word_vec_num]

            totalVec = np.concatenate((node_vec,word_vec))
            changeList.append(totalVec)

        if len(changeList) < flag.node_num:
            for i in range(flag.node_num-len(changeList)):
                changeList.append(np.zeros(flag.total_dim))
        changeList = changeList[:flag.node_num]
        data_list.append(changeList)
        answer_list.append(node.is_buggy)
        if idx % 100 == 0:
            print("read ",idx)

    batch_manager = utils_new.BatchManager(x=data_list, y=answer_list, batch_size=flag.batch_size, valid_ratio=0.2)
    only_once = True
    for i in range(5):
        print("training with fold index : ",i)
        print("start train in 3 sec...")
        time.sleep(3)
        batch_manager.cross_setting(i)
        with tf.Session() as sess:
            train_model = model_new.Classifier()
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(flag.tensorboard_url + "/train_fold_" + str(i))
            valid_writer = tf.summary.FileWriter(flag.tensorboard_url + "/valid_fold_" + str(i))
            for cur_epoch in range(flag.epochs):
                start = time.time()
                for batch_x, batch_y in batch_manager.get_batches():
                    batch_y = np.reshape(batch_y, [-1, 1])
                    current_step = tf.train.global_step(sess, train_model.global_step)
                    feed = {train_model.X: batch_x, train_model.Y: batch_y, train_model.dropout_rate: 0.3}
                    _, step, cost, accuracy = sess.run(
                        [train_model.train_op, train_model.global_step, train_model.cost, train_model.accuracy],
                        feed_dict=feed)
                    if current_step % 10 == 0:
                        cost_summ, accuracy_summ = sess.run([train_model.cost_summ, train_model.accuracy_summ],
                                                            feed_dict=feed)
                        train_writer.add_summary(cost_summ, current_step)
                        train_writer.add_summary(accuracy_summ, current_step)
                    print("{} ({} epochs) step, loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
                          .format(current_step, cur_epoch, cost, accuracy, time.time() - start))

                    if current_step % 100 == 0:
                        avg_loss, avg_accuracy = 0.0, 0.0
                        started = time.time()
                        for valid_x, valid_y in batch_manager.get_valid_batches():
                            valid_y = np.reshape(valid_y, [-1, 1])
                            feed = {train_model.X: valid_x, train_model.Y: valid_y, train_model.dropout_rate: 0.0}
                            loss, accuracy, cost_summ, accuracy_summ, gbstep = sess.run(
                                [train_model.cost, train_model.accuracy, train_model.cost_summ,
                                 train_model.accuracy_summ, train_model.global_step], feed_dict=feed)
                            avg_accuracy += accuracy * len(valid_x)
                            avg_loss += loss * len(valid_x)

                        print("({} epochs) evaluation step, loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
                              .format(cur_epoch, avg_loss / len(batch_manager.valid_x),
                                      avg_accuracy / len(batch_manager.valid_x), time.time() - started))
                        summ = tf.Summary()
                        summ.value.add(tag="accuracy", simple_value=avg_accuracy / len(batch_manager.valid_x))
                        valid_writer.add_summary(summ, gbstep)
                        summ2 = tf.Summary()
                        summ2.value.add(tag="cost", simple_value=avg_loss / len(batch_manager.valid_x))
                        valid_writer.add_summary(summ2, gbstep)
                    start = time.time()
        if only_once:
            break

#main("")
