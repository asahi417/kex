""" UnitTest for phrase_constructor"""
import unittest
import logging

import kex

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


test_en = 'Efficient discovery of grid services is essential for the success of grid computing. The ' \
           'standardization of grids based on web services has resulted in the need for scalable web service ' \
           'discovery mechanisms to be deployed in grids Even though UDDI has been the de facto industry standard ' \
           'for web-services discovery, imposed requirements of tight-replication among registries and lack of ' \
           'autonomous control has severely hindered its widespread deployment and usage. With the advent of grid ' \
           'computing the scalability issue of UDDI will become a -lsb- roadblock that will prevent its deployment in ' \
           'grids. In this paper we present our distributed web-service discovery architecture, called DUDE ' \
           '(Distributed UDDI Deployment Engine). DUDE leverages DHT -LSB- (Distributed Hash Tables) as a rendezvous ' \
           'mechanism between multiple UDDI registries. DUDE enables consumers to query multiple registries, ' \
           'still at the same time allowing organizations to have autonomous control over their registries. ' \
           'Based on preliminary prototype on PlanetLab, we believe that DUDE architecture can support effective ' \
           'distribution of UDDI registries thereby making UDDI more robust and also addressing its scaling ' \
           'issues. Furthermore, The DUDE architecture for scalable distribution can be applied beyond UDDI to ' \
           'any Grid Service Discovery mechanism.'


class CustomExtractor:
    """ Custom keyword extractor example: First N keywords extractor """

    def __init__(self, maximum_word_number: int = 3):
        """ First N keywords extractor """
        self.phrase_constructor = kex.PhraseConstructor(maximum_word_number=maximum_word_number)

    def get_keywords(self, document: str, n_keywords: int = 10):
        """ Get keywords

         Parameter
        ------------------
        document: str
        n_keywords: int

         Return
        ------------------
        a list of dictionary consisting of 'stemmed', 'pos', 'raw', 'offset', 'count'.
        eg) {'stemmed': 'grid comput', 'pos': 'ADJ NOUN', 'raw': ['grid computing'], 'offset': [[11, 12]], 'count': 1}
        """
        phrase_instance, stemmed_tokens = self.phrase_constructor.tokenize_and_stem_and_phrase(document)
        sorted_phrases = sorted(phrase_instance.values(), key=lambda x: x['offset'][0][0])
        return sorted_phrases[:min(len(sorted_phrases), n_keywords)]


class TestCustomExtractor(unittest.TestCase):
    """Test custom extractor """

    def test(self):
        model = CustomExtractor()
        out = model.get_keywords(test_en)
        logging.info(out)


if __name__ == "__main__":
    unittest.main()
