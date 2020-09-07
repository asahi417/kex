""" UnitTest for dataset """
import unittest
import logging
from logging.config import dictConfig

from grapher import get_benchmark_dataset

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()


class TestGetDataset(unittest.TestCase):
    """Test get_benchmark_dataset """

    def test_get_benchmark_dataset(self):
        tmp = get_benchmark_dataset()
        for n, (k, v) in enumerate(tmp.items()):
            LOGGER.info('\n - {0}: \n * source: {1} \n * keywords: {2}'.format(k, v['source'], v['keywords']))
            if n > 5:
                break


if __name__ == "__main__":
    unittest.main()
