from setuptools import setup, find_packages

VERSION = '0.0.0'
NAME = 'kex'

with open('README.md') as f:
    readme = f.read()

setup(
    name=NAME,
    packages=[NAME],
    version=VERSION,
    license='MIT',
    description='Light/easy keyword extraction.',
    url='https://github.com/asahi417/kex',
    download_url='https://github.com/user/reponame/archive/v_01.tar.gz',  # I explain this later on
    keywords=['keyword-extraction', 'nlp', 'information-retrieval'],
    long_description=readme,
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
      ],
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