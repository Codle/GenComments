""" Main Entery
"""

import argparse
import importlib

import tensorflow as tf
import texar.tf as tx
from configs import data as config_data
from configs import transformer as config_model
from models import transformer, seq2seq
from engine import trainer

parser = argparse.ArgumentParser()
# TODO: 参数列表


def main():
    """程序入口
    """
    train_data = tx.data.PairedTextData(hparams=config_data.train)
    valid_data = tx.data.PairedTextData(hparams=config_data.valid)
    test_data = tx.data.PairedTextData(hparams=config_data.test)

    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=valid_data, test=test_data)

    batch = data_iterator.get_next()

    # 构建模型
    train_op, _ = transformer.build_model(
        batch, train_data, config_data, config_model)

    # 启动训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        trainer._train_epoch(sess, train_op, data_iterator, config_data)
    #     best_val_bleu = -1.
    #     for i in range(config_data.num_epochs):
    #         _train_epoch(sess)

    #         val_bleu = _eval_epoch(sess, 'val')
    #         best_val_bleu = max(best_val_bleu, val_bleu)
    #         print('val epoch={}, BLEU={:.4f}; best-ever={:.4f}'.format(
    #             i, val_bleu, best_val_bleu))

    #         test_bleu = _eval_epoch(sess, 'test')
    #         print('test epoch={}, BLEU={:.4f}'.format(i, test_bleu))

    #         print('=' * 50)


if __name__ == '__main__':
    main()
