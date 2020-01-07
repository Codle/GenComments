""" Main Entery
"""
import os
import tensorflow as tf
import texar.tf as tx
from configs import data as config_data
from configs import transformer as config_model
from models import transformer
from utils import utils
from bleu_tool import bleu_wrapper

flags = tf.flags

flags.DEFINE_string("run_mode", "train_and_evaluate",
                    "Either train_and_evaluate or test.")
flags.DEFINE_string("model_dir", "./outputs",
                    "Directory to save the trained model and logs.")

FLAGS = flags.FLAGS


def main():
    """程序入口
    """
    # Create logging
    tx.utils.maybe_create_dir(FLAGS.model_dir)
    logging_file = os.path.join(FLAGS.model_dir, 'logging.txt')
    logger = utils.get_logger(logging_file)
    print('logging file is saved in: %s', logging_file)

    # 读取数据
    train_data = tx.data.PairedTextData(hparams=config_data.train)
    valid_data = tx.data.PairedTextData(hparams=config_data.valid)
    test_data = tx.data.PairedTextData(hparams=config_data.test)

    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=valid_data, test=test_data)

    batch = data_iterator.get_next()

    # 构建模型
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')

    train_op, beam_search_ids, mle_loss, summary_merged = transformer.build_model(
        batch,
        train_data,
        config_data,
        config_model,
        global_step,
        learning_rate)

    saver = tf.train.Saver(max_to_keep=5)
    best_results = {'score': 0, 'epoch': -1}

    def _eval_epoch(sess, epoch, mode):
        if mode == 'eval':
            data_iterator.switch_to_val_data(sess)
        elif mode == 'test':
            data_iterator.switch_to_test_data(sess)
        else:
            raise ValueError('`mode` should be either "eval" or "test".')

        references, hypotheses = [], []
        while True:
            try:
                feed_dict = {
                    tx.global_mode(): tf.estimator.ModeKeys.EVAL,
                }
                fetches = {
                    'beam_search_ids': beam_search_ids,
                }
                fetches_ = sess.run(fetches, feed_dict=feed_dict)

                hypotheses.extend(h.tolist()
                                  for h in fetches_['beam_search_ids'])
                references.extend(r.tolist() for r in batch['target_text_ids'])
                hypotheses = utils.list_strip_eos(
                    hypotheses, train_data.target_vocab.eos_token_id)
                references = utils.list_strip_eos(
                    references, train_data.target_vocab.eos_token_id)
            except tf.errors.OutOfRangeError:
                break

        if mode == 'eval':
            # Writes results to files to evaluate BLEU
            # For 'eval' mode, the BLEU is based on token ids (rather than
            # text tokens) and serves only as a surrogate metric to monitor
            # the training process
            fname = os.path.join(FLAGS.model_dir, 'tmp.eval')
            hypotheses = tx.utils.str_join(hypotheses)
            references = tx.utils.str_join(references)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hypotheses, references, fname, mode='s')
            eval_bleu = bleu_wrapper(ref_fn, hyp_fn, case_sensitive=True)
            eval_bleu = 100. * eval_bleu
            logger.info('epoch: %d, eval_bleu %.4f', epoch, eval_bleu)
            print('epoch: %d, eval_bleu %.4f' % (epoch, eval_bleu))

            if eval_bleu > best_results['score']:
                logger.info('epoch: %d, best bleu: %.4f', epoch, eval_bleu)
                best_results['score'] = eval_bleu
                best_results['epoch'] = epoch
                model_path = os.path.join(FLAGS.model_dir, 'best-model.ckpt')
                logger.info('saving model to %s', model_path)
                print('saving model to %s' % model_path)
                saver.save(sess, model_path)

        elif mode == 'test':
            # For 'test' mode, together with the cmds in README.md, BLEU
            # is evaluated based on text tokens, which is the standard metric.
            fname = os.path.join(FLAGS.model_dir, 'test.output')
            hwords, rwords = [], []
            for hyp, ref in zip(hypotheses, references):
                hword = train_data.target_vocab.map_ids_to_tokens(hyp)
                hwords.append(hword)
                rword = train_data.target_vocab.map_ids_to_tokens(ref)
                rwords.append(rword)
            hwords = tx.utils.str_join(hwords)
            rwords = tx.utils.str_join(rwords)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hwords, rwords, fname, mode='s',
                src_fname_suffix='hyp', tgt_fname_suffix='ref')
            logger.info('Test output writtn to file: %s', hyp_fn)
            print('Test output writtn to file: %s' % hyp_fn)

    def _train_epoch(sess, epoch, step, smry_writer):
        data_iterator.switch_to_train_data(sess)
        while True:
            try:
                feed_dict = {
                    learning_rate: utils.get_lr(step, config_model.lr)
                }
                fetches = {
                    'step': global_step,
                    'train_op': train_op,
                    'smry': summary_merged,
                    'loss': mle_loss,
                }

                fetches_ = sess.run(fetches, feed_dict=feed_dict)
                step, loss = fetches_['step'], fetches_['loss']
                if step and step % config_data.display_steps == 0:
                    logger.info('step: %d, loss: %.4f', step, loss)
                    print('step: %d, loss: %.4f' % (step, loss))
                    smry_writer.add_summary(fetches_['smry'], global_step=step)
                if step and step % config_data.eval_steps == 0:
                    _eval_epoch(sess, epoch, mode='eval')
                    _eval_epoch(sess, epoch, mode='test')
            except tf.errors.OutOfRangeError:
                break
        return step

    # 启动训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        smry_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        if FLAGS.run_mode == 'train_and_evaluate':
            logger.info('Begin running with train_and_evaluate mode')

            if tf.train.latest_checkpoint(FLAGS.model_dir) is not None:
                logger.info('Restore latest checkpoint in %s' %
                            FLAGS.model_dir)
                saver.restore(
                    sess, tf.train.latest_checkpoint(FLAGS.model_dir))

            step = 0
            for epoch in range(config_data.max_train_epoch):
                step = _train_epoch(sess, epoch, step, smry_writer)


if __name__ == '__main__':
    main()
