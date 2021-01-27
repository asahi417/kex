import kex


VALID_ALGORITHMS = [
    'FirstN', 'TF', 'LexSpec', 'TFIDF', 'TextRank', 'SingleRank', 'PositionRank', 'LexRank', 'TFIDFRank', 'SingleTPR',
    'TopicRank'
]


def get_algorithm(model_name, *args, **kwargs):

    if model_name == 'TopicRank':
        model = kex.TopicRank(*args, **kwargs)
    elif model_name == 'TextRank':
        model = kex.TextRank(*args, **kwargs)
    elif model_name == 'SingleRank':
        model = kex.SingleRank(*args, **kwargs)
    elif model_name == 'TFIDFRank':
        model = kex.TFIDFRank(*args, **kwargs)
    elif model_name == 'PositionRank':
        model = kex.PositionRank(*args, **kwargs)
    elif model_name == 'TFIDF':
        model = kex.TFIDF(*args, **kwargs)
    elif model_name == 'LexSpec':
        model = kex.LexSpec(*args, **kwargs)
    elif model_name == 'LexRank':
        model = kex.LexRank(*args, **kwargs)
    elif model_name == 'FirstN':
        model = kex.FirstN(*args, **kwargs)
    elif model_name == 'TF':
        model = kex.TF(*args, **kwargs)
    elif model_name == 'TopicalPageRank':
        model = kex.TopicalPageRank(*args, **kwargs)
    elif model_name == 'SingleTPR':
        model = kex.SingleTPR(*args, **kwargs)
    else:
        raise ValueError('unknown model: {}\n valid algorithms are: {}'.format(model_name, VALID_ALGORITHMS))

    return model
