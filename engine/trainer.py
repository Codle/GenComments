from . import inference
import tensorflow as tf


def _train_epoch(sess,
                 train_op,
                 data_iterator,
                 config_data):
    """ 训练一个 epoch

    参数:
        sess: session
        data_iterator: 数据迭代器
        config_data: 配置文件
        smry_writer: Tensorboard 日志
    返回值:

    """
    data_iterator.switch_to_train_data(sess)
    step = 0
    while True:
        try:
            loss = sess.run(train_op)
            if step % config_data.display == 0:
                print("step={}, loss={:.4f}".format(step, loss))
            # if (config_data.train_with_eval and
            #         step % config_data.eval_step == 0):
            #     interfence.do_eval()
            step += 1
        except tf.errors.OutOfRangeError:
            break


def do_train(sess, train_op, data_iterator, config_data, smry_writer):
    for i in range(config_data.num_epoch):
        step = 0
        _train_epoch(sess, step, data_iterator, config_data, smry_writer)
