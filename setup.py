from setuptools import setup, find_packages

VERSION = '0.0.0'
NAME = 'grapher'
IS_RELEASED = False

with open('README.md') as f:
    readme = f.read()

setup(
    name=NAME,
    version=VERSION,
    description='Graph-based keyword extractions.',
    long_description=readme,
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    packages=find_packages(exclude=('random', 'dataset')),
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        'gensim>=3.4.0,<3.5.0',
        'nltk==3.4.1',
        'spacy==2.1.4',
        'networkx==2.3',
        'numpy>=1.16.1',
        'mecab-python3==0.996.2',
        'untangle'
    ]
)
