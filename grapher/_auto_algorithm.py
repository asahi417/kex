import grapher


VALID_ALGORITHMS = ['FirstN', 'TF', 'LexSpec', 'TFIDF', 'TextRank', 'SingleRank', 'PositionRank', 'LexRank',
                    'ExpandRank', 'SingleTPR', 'TopicRank']


def AutoAlgorithm(model_name, *args, **kwargs):

    if model_name == 'TopicRank':
        model = grapher.TopicRank(*args, **kwargs)
    elif model_name == 'TextRank':
        model = grapher.TextRank(*args, **kwargs)
    elif model_name == 'SingleRank':
        model = grapher.SingleRank(*args, **kwargs)
    elif model_name == 'ExpandRank':
        model = grapher.ExpandRank(*args, **kwargs)
    elif model_name == 'PositionRank':
        model = grapher.PositionRank(*args, **kwargs)
    elif model_name == 'TFIDF':
        model = grapher.TFIDF(*args, **kwargs)
    elif model_name == 'LexSpec':
        model = grapher.LexSpec(*args, **kwargs)
    elif model_name == 'LexRank':
        model = grapher.LexRank(*args, **kwargs)
    elif model_name == 'FirstN':
        model = grapher.FirstN(*args, **kwargs)
    elif model_name == 'TF':
        model = grapher.TF(*args, **kwargs)
    # elif model_name == 'MultipartiteRank':
    #     model = grapher.MultipartiteRank(*args, **kwargs)
    # elif model_name == 'TopicalPageRank':
    #     model = grapher.TopicalPageRank(*args, **kwargs)
    elif model_name == 'SingleTPR':
        model = grapher.SingleTPR(*args, **kwargs)
    else:
        raise ValueError('unknown model: {}\n valid algorithms are: {}'.format(model_name, VALID_ALGORITHMS))

    return model
