from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

VERSION = '2.0.5'
setup(
    name='kex',
    packages=find_packages(exclude=['examples', 'tests', 'asset', 'benchmark']),
    version=VERSION,
    license='MIT',
    description='Light/easy keyword extraction from documents.',
    url='https://github.com/asahi417/kex',
    download_url="https://github.com/asahi417/kex/archive/v{}.tar.gz".format(VERSION),
    keywords=['keyword-extraction', 'nlp', 'information-retrieval'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
      ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        'gensim>=3.4.0,<3.5.0',
        'nltk==3.5',
        'networkx',
        'numpy>=1.16.1',
        'segtok',
        'requests'
    ],
    python_requires='>=3.6',
)
