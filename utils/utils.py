import logging
import math


def get_lr(fstep, opt_config):
    if opt_config['learning_rate_schedule'] == 'static':
        lr = opt_config['static_lr']
    else:
        lr = opt_config['lr_constant'] \
            * min(1.0, (fstep / opt_config['warmup_steps'])) \
            * (1 / math.sqrt(max(fstep, opt_config['warmup_steps'])))
    return lr


def get_logger(log_path):
    """Returns a logger.
    Args:
        log_path (str): Path to the log file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
    logger.addHandler(fh)
    return logger


def list_strip_eos(list_, eos_token):
    """Strips EOS token from a list of lists of tokens.
    """
    list_strip = []
    for elem in list_:
        if eos_token in elem:
            elem = elem[:elem.index(eos_token)]
        list_strip.append(elem)
    return list_strip
