from . import graph, stat

ALGORITHM_SET = {
    'TextRank': graph.TextRank,
    'SingleRank': graph.SingleRank,
    'TopicRank': graph.TopicRank,
    # 'ExpandRank': graph.ExpandRank,
    # 'TopicalPageRank': graph.TopicalPageRank,
    # 'SingleTPR': graph.SingleTPR,
    # 'MultipartiteRank': graph.MultipartiteRank,
    # 'PositionRank': graph.PositionRank,
    'TFIDFBased': stat.TFIDFBased
}


def algorithm_instance(algorithm_name: str, **kwargs):
    if algorithm_name not in ALGORITHM_SET.keys():
        raise ValueError('unknown algorithm: %s not in %s' % (algorithm_name, ALGORITHM_SET.keys()))
    instance = ALGORITHM_SET[algorithm_name]
    if len(kwargs) > 0:
        return instance(**kwargs)
    else:
        return instance()
