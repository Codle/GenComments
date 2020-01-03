"""Data Configure
"""

num_epoch = 10
display = 5

source_vocab_file = 'data/output/vocab.titles'
target_vocab_file = 'data/output/vocab.comments'

train = {
    'batch_size': 100,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        'files': 'data/processed/train_bpe_title',
        'vocab_file': source_vocab_file,
        'max_seq_length': 150
    },
    'target_dataset': {
        'files': 'data/processed/train_bpe_comment',
        'vocab_file': target_vocab_file,
        'max_seq_length': 150
    }
}

valid = {
    'batch_size': 100,
    'shuffle': False,
    'source_dataset': {
        'files': 'data/processed/valid_bpe_title',
        'vocab_file': source_vocab_file
    },
    'target_dataset': {
        'files': 'data/processed/valid_bpe_comment',
        'vocab_file': target_vocab_file
    }
}

test = {
    'batch_size': 100,
    'shuffle': False,
    'source_dataset': {
        'files': 'data/processed/test_bpe_title',
        'vocab_file': source_vocab_file
    },
    'target_dataset': {
        'files': 'data/processed/test_bpe_comment',
        'vocab_file': target_vocab_file
    }
}
