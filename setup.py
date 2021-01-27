from setuptools import setup, find_packages

VERSION = '0.0.0'
NAME = 'kex'
IS_RELEASED = False

with open('README.md') as f:
    readme = f.read()

setup(
    name=NAME,
    version=VERSION,
    description='Light/easy keyword extraction.',
    long_description=readme,
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        'gensim>=3.4.0,<3.5.0',
        'nltk==3.5',
        'spacy==2.3.2',
        'networkx',
        'numpy>=1.16.1',
        'untangle',
        'tqdm',
        'segtok',
        'pandas'
    ]
)
