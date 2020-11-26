""" UnitTest for phrase_constructor"""
import unittest
import logging

from grapher import PhraseConstructor

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


class TestPhraseConstructor(unittest.TestCase):
    """Test phrase_constructor """

    def test_tokenize_and_stem_and_phrase(self):
        phraser = PhraseConstructor()
        phrase, token = phraser.tokenize_and_stem_and_phrase(test_en)
        logging.info(token)
        for k, v in phrase.items():
            assert 'NOUN' in v['pos'] or 'PROPN' in v['pos'], 'noun not found {}'.format(v['pos'])
        logging.info(phrase.keys())

    def test_tokenize_and_stem(self):
        phraser = PhraseConstructor()
        logging.info(phraser.tokenize_and_stem(test_en))
        logging.info(phraser.tokenize_and_stem(test_en, apply_stopwords=False))


if __name__ == "__main__":
    unittest.main()
