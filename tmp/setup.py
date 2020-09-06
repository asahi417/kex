from setuptools import setup, find_packages
from simple_versioner import write_version_py


VERSION = '0.2.0'
NAME = 'keyphraser'
IS_RELEASED = False

FULL_VERSION = write_version_py(NAME, VERSION, is_released=IS_RELEASED)

with open('README.md') as f:
    readme = f.read()

setup(
    name=NAME,
    version=FULL_VERSION,
    description='Collection of automatic keyphrase extraction algorithms',
    long_description=readme,
    author='Asahi Ushio',
    author_email='aushio@cogent.co.jp',
    packages=find_packages(exclude=('random', 'dataset')),
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        'gensim==3.7.3',  # 3.4.0
        'nltk==3.4.1',
        'untangle==1.1.1',  # parsing xml file
        'spacy==2.1.4',
        'networkx==2.3',
        'numpy==1.16.3',
        'pandas==0.24.2',
        'mecab-python3==0.996.2',
    ]
)
