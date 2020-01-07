""" 生成 BERT 的 TFRecord
"""

import os
import csv
import collections

import tensorflow as tf

import texar.tf as tx


class InputExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, source, target):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
                For single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
                sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.source = source
        self.target = target


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, target_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.target_ids = target_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        lines = []
        with tf.gfile.Open(input_file, "r") as f:
            for line in f.readlines():
                lines.append(line.strip())
        return lines


class BERTProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        src_lines = self._read_tsv(os.path.join(data_dir, "train_title"))
        tgt_lines = self._read_tsv(os.path.join(data_dir, "train_comment"))
        return self._create_examples(
            zip(src_lines, tgt_lines), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        src_lines = self._read_tsv(os.path.join(data_dir, "val_title"))
        tgt_lines = self._read_tsv(os.path.join(data_dir, "val_comment"))
        return self._create_examples(
            zip(src_lines, tgt_lines), "val")

    def get_test_examples(self, data_dir):
        """See base class."""
        src_lines = self._read_tsv(os.path.join(data_dir, "test_title"))
        tgt_lines = self._read_tsv(os.path.join(data_dir, "test_comment"))
        return self._create_examples(
            zip(src_lines, tgt_lines), "test")

    @staticmethod
    def _create_examples(zip_lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(zip_lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            source = tx.utils.compat_as_text(line[0])
            target = tx.utils.compat_as_text(line[1])
            examples.append(InputExample(guid=guid, source=source,
                                         target=target))
        return examples


def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    input_ids, segment_ids, input_mask = \
        tokenizer.encode_text(text_a=example.source,
                              max_seq_length=max_seq_length)

    target_ids, _, _ = tokenizer.encode_text(text_a=example.target,
                                       max_seq_length=max_seq_length)

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            target_ids=target_ids)
    return feature


def convert_examples_to_features_and_output_to_files(
        examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):

        feature = convert_single_example(ex_index, example,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["target_ids"] = create_int_feature(feature.target_ids)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def prepare_TFRecord_data(processor, tokenizer,
                          data_dir, max_seq_length, output_dir):
    """
    Args:
        processor: Data Preprocessor, which must have get_lables,
            get_train/dev/test/examples methods defined.
        tokenizer: The Sentence Tokenizer. Generally should be
            SentencePiece Model.
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        output_dir: The directory to save the TFRecord in.
    """

    train_examples = processor.get_train_examples(data_dir)
    train_file = os.path.join(output_dir, "train.tf_record")
    convert_examples_to_features_and_output_to_files(
        train_examples, max_seq_length, tokenizer, train_file)

    eval_examples = processor.get_dev_examples(data_dir)
    eval_file = os.path.join(output_dir, "val.tf_record")
    convert_examples_to_features_and_output_to_files(
        eval_examples, max_seq_length, tokenizer, eval_file)

    test_examples = processor.get_test_examples(data_dir)
    test_file = os.path.join(output_dir, "test.tf_record")
    convert_examples_to_features_and_output_to_files(
        test_examples, max_seq_length, tokenizer, test_file)


def main():
    processor = BERTProcessor()
    prepare_TFRecord_data(
        processor,
        tx.data.BERTTokenizer('bert-base-chinese'),
        'data/processed',
        150,
        'data/bert')


if __name__ == '__main__':
    main()
