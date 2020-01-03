"""创建 Transformer 模型
"""
import texar.tf as tx
import tensorflow as tf
from texar.tf.utils import transformer_utils


def build_model(batch, train_data, config_data, config_model):
    """创建 Transformer 模型

    参数:
        batch: 一个批次训练数据
        train_data: 整个训练数据，用于获取一些基本信息

    返回值:
        train_op: 训练运算符
        beam_search_outputs: BeamSearch 结果
    """
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

    # 源嵌入
    src_word_embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.source_vocab.size, hparams=config_model.emb)
    src_word_embeds = src_word_embedder(batch['source_text_ids'])
    src_word_embeds = src_word_embeds * config_model.hidden_dim ** 0.5

    # 位置编码 (源和目标共享)
    pos_embedder = tx.modules.SinusoidsPositionEmbedder(
        position_size=config_data.train['target_dataset']['max_seq_length']+1,
        hparams=config_model.position_embedder_hparams)
    src_pos_embeds = pos_embedder(sequence_length=batch['source_length'])

    src_input_embedding = src_word_embeds + src_pos_embeds

    encoder = tx.modules.TransformerEncoder(hparams=config_model.encoder)
    encoder_output = encoder(inputs=src_input_embedding,
                             sequence_length=batch['source_length'])

    # 目标嵌入，显式增加 <PAD>
    tgt_embedding = tf.concat(
        [tf.zeros(shape=[1, src_word_embedder.dim]),
         src_word_embedder.embedding[1:, :]],
        axis=0)
    tgt_embedder = tx.modules.WordEmbedder(tgt_embedding)

    tgt_word_embeds = tgt_embedder(batch['target_text_ids'])
    tgt_word_embeds = tgt_word_embeds * config_model.hidden_dim ** 0.5

    tgt_pos_embeds = pos_embedder(sequence_length=batch['target_length'])
    tgt_input_embedding = tgt_word_embeds + tgt_pos_embeds

    _output_w = tf.transpose(tgt_embedder.embedding, (1, 0))

    decoder = tx.modules.TransformerDecoder(
        vocab_size=train_data.target_vocab.size,
        output_layer=_output_w,
        hparams=config_model.decoder)

    # 用于训练
    outputs = decoder(
        memory=encoder_output,
        memory_sequence_length=batch['source_length'],
        inputs=tgt_input_embedding,
        decoding_strategy='train_greedy',
        mode=tf.estimator.ModeKeys.TRAIN
    )

    # mle_loss = transformer_utils.smoothing_cross_entropy(
    #     logits=outputs.logits,
    #     labels=batch['target_text_ids'][:, 1:],
    #     vocab_size=train_data.target_vocab.size,
    #     confidence=config_model.loss_label_confidence)
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=batch['target_text_ids'][:, 1:],
        logits=outputs.logits,
        sequence_length=batch['target_length'] - 1)

    # 不清楚干嘛的先不用
    # mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)

    train_op = tx.core.get_train_op(
        mle_loss,
        learning_rate=config_model.learning_rate,
        global_step=global_step,
        hparams=config_model.opt)

    # 用于推理 (beam-search)
    start_tokens = tf.ones_like(batch['target_length']) * \
        train_data.target_vocab.bos_token_id

    # 应用了位置编码的嵌入表示
    def _embedding_fn(x, y):
        x_w_embed = tgt_embedder(x)
        y_p_embed = pos_embedder(y)
        return x_w_embed * config_model.hidden_dim ** 0.5 + y_p_embed

    predictions = decoder(
        memory=encoder_output,
        memory_sequence_length=batch['source_length'],
        beam_width=config_model.beam_width,
        length_penalty=config_model.length_penalty,
        start_tokens=start_tokens,
        end_token=train_data.target_vocab.eos_token_id,
        embedding=_embedding_fn,
        max_decoding_length=config_data.train['target_dataset']['max_seq_length'],
        mode=tf.estimator.ModeKeys.PREDICT)
    # Uses the best sample by beam search
    beam_search_ids = predictions['sample_id'][:, :, 0]

    return train_op, beam_search_ids
