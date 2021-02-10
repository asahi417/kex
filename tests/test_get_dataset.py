""" UnitTest for dataset """
import unittest
import logging

from kex import get_benchmark_dataset, VALID_DATASET_LIST

LOGGER = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class TestGetDataset(unittest.TestCase):
    """Test get_benchmark_dataset """

    def test_get_benchmark_dataset(self):
        for i in VALID_DATASET_LIST:
            LOGGER.info('** {} **'.format(i))
            tmp, language = get_benchmark_dataset(i)
            LOGGER.info(language)
            for n, v in enumerate(tmp):
                LOGGER.info('\n - {0}: \n * source: {1} \n * keywords: {2}'.format(v['id'], v['source'], v['keywords']))
                if n > 5:
                    break


if __name__ == "__main__":
    unittest.main()
