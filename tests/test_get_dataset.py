""" UnitTest for dataset """
import unittest
import logging

from grapher import get_benchmark_dataset, VALID_DATASET_LIST, get_statistics

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

    def test_statistics(self):
        tmp, language = get_benchmark_dataset('Inspec')
        for n, i in enumerate(tmp):
            logging.info(i['source'])
            out = get_statistics(i['keywords'], i['source'])
            logging.info('- invalid: {}'.format(out['label_invalid']))
            logging.info('- valid  : {}'.format(out['label_valid']))
            input()
            if n > 10:
                break


if __name__ == "__main__":
    unittest.main()
