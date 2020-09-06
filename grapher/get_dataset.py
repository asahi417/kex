import os

CACHE_DIR = './cache'
__all__ = 'get_benchmark_dataset'


def get_benchmark_dataset(cache_dir: str = None):
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists('{}/test.combined.stem.final'.format(cache_dir)):
        os.system('wget -O {}/SemEval2010.tar.gz https://github.com/snkim/AutomaticKeyphraseExtraction/raw/master/SemEval2010.tar.gz'.
                  format(cache_dir))
        os.system('tar -xzf {0}/SemEval2010.tar.gz -C {0}'.format(cache_dir))
        os.system('tar -xzf {0}/SemEval2010/test_answer.tar.gz -C {0}'.format(cache_dir))
        os.system('tar -xzf {0}/SemEval2010/test.tar.gz -C {0}'.format(cache_dir))
    with open('{}/test.combined.stem.final'.format(cache_dir), 'r') as f:
        answer_dict = {
            i.split(' : ')[0]: {
                'source_file': open('{0}/test/{1}.txt.final'.format(cache_dir, i.split(' : ')[0]), 'r').read(),
                'text': i.split(' : ')[1]
            } for i in f.read().split('\n') if len(i) > 0}
    return answer_dict




