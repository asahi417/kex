""" UnitTest for TextRank """
import unittest
import logging
from logging.config import dictConfig

from grapher import TopicRank, MultipartiteRank

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()


class TestTopicRank(unittest.TestCase):
    """Test graph_topic_rank"""

    def test_text_rank(self):
        model = TopicRank(language='ja')
        test = '東京事変は、2003年から活動している日本のバンドである。2012年2月29日の日本武道館公演をもって活動を終了したが、' \
               'その後2020年に「再生」と称して解散時のメンバーで再始動することが発表された。所属レコード会社はユニバーサルミュージック。' \
               'シンガーソングライターの椎名林檎を中心に、2003年に結成された5人組のロックバンド。2012年2月29日の日本武道館公演をもって' \
               '活動を終了。その後、2020年に「再生」と称して解散時のメンバーで再始動することが発表された。'
        LOGGER.info(test)
        out = model.get_keywords(test)
        for i in out:
            LOGGER.info(i)


if __name__ == "__main__":
    unittest.main()
