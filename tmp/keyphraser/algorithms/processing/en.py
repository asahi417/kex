""" Preprocess raw document to get sequence of phrase and token [English] """

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex
from nltk import SnowballStemmer
import re
from .basic_modules import CleaningStr, CandidatePhrase


class Processing:
    """ Pipeline of cleaning filter, spacy object to get nlp attribution (stem, lemma, pos), and phrasing.

     Usage
    -----------------
    >>> sample =
    'We propose a new evaluation strategy for keyphrase extraction based on approximate keyphrase matching.
    It corresponds well with human judgments and is better suited to assess the performance of keyphrase extraction
    approaches. Additionally, we propose a generalized framework for comprehensive analysis of keyphrase extraction
    that subsumes most existing approaches, which allows for fair testing conditions. For the first time, we compare the
    results of state-of-the-art unsupervised and supervised keyphrase extraction approaches on three evaluation datasets
    and show that the relative performance of the approaches heavily depends on the evaluation metric as well as on the
    properties of the evaluation dataset.'
    >>> processor = Processing()
    >>> result_dict = processor(sample)
    >>> result_dict.keys()
        dict_keys(['new evalu strategi', 'keyphras extract', 'approxim keyphras match', 'evalu dataset'])
    >>> result_dict['evalu dataset']
        {
            'count': 2,
            'stemmed': 'evalu dataset',
            'pos': 'NOUN NOUN',
            'raw': ['evaluation datasets', 'evaluation dataset'],
            'lemma': ['evaluation dataset', 'evaluation dataset'],
            'offset': [[86, 87], [111, 112]]
        }


     Reference
    ------------------
    * eos detection by spacy: not needed for keyphrase extraction but might be needed in sentence-wise processing
                              https://github.com/explosion/spaCy/issues/23
    """

    def __init__(self,
                 add_verb: bool = False):

        # cleaning filter instance
        self.__clean_str = CleaningStr(language='en')

        # setup spacy nlp model
        self.__spacy_processor = self.setup_spacy_processor()
        self.__stemming = SnowballStemmer('english')

        # candidate phrasing
        self.__add_verb = add_verb

    @staticmethod
    def setup_spacy_processor():

        try:
            spacy_processor = spacy.load('en_core_web_sm')
        except OSError:
            from spacy.cli import download
            download('en_core_web_sm')
            spacy_processor = spacy.load('en_core_web_sm')

        def custom_tokenizer():
            # avoid splitting by `-`: ex) `off-site apple store` -> ['off-site', 'apple', 'store']
            # avoid splitting by `.`: ex) `U.S. and U.K.` -> ['U.S.', 'and', 'U.K.']
            # infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
            infix_re = re.compile(r'''[\,\?\:\;\‘\’\`\“\”\"\'~]''')
            prefix_re = compile_prefix_regex(spacy_processor.Defaults.prefixes)
            suffix_re = compile_suffix_regex(spacy_processor.Defaults.suffixes)

            return Tokenizer(spacy_processor.vocab,
                             prefix_search=prefix_re.search,
                             suffix_search=suffix_re.search,
                             infix_finditer=infix_re.finditer,
                             token_match=None)

        spacy_processor.tokenizer = custom_tokenizer()
        return spacy_processor

    def __call__(self,
                 single_doc: str,
                 return_token: bool = False):

        phrase_instance = CandidatePhrase(add_verb=self.__add_verb)
        cleaned_str = self.__clean_str.process(single_doc)
        cleaned_str_tokenize = []
        n = 0
        for n, spacy_object in enumerate(self.__spacy_processor(cleaned_str)):
            cleaned_str_tokenize.append(spacy_object.text)
            phrase_instance.add(raw=spacy_object.text,
                                lemma=spacy_object.lemma_,
                                stemmed=self.__stemming.stem(spacy_object.text),
                                pos=spacy_object.pos_,
                                offset=n)

        # to finalize the phrase in the end of sentence
        phrase_instance.add(raw='.',
                            lemma='.',
                            stemmed='.',
                            pos='PUNCT',
                            offset=n+1)

        if return_token:
            return phrase_instance.phrase, cleaned_str_tokenize
        else:
            return phrase_instance.phrase

    @property
    def spacy_processor(self):
        return self.__spacy_processor

    @property
    def stemmer(self):
        return self.__stemming.stem
