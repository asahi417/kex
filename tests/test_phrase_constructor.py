""" UnitTest for phrase_constructor"""
import unittest
import logging
from logging.config import dictConfig

from grapher import PhraseConstructor

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()

test_ja = '東京事変は、2003年から活動している日本のバンドである。2012年2月29日の日本武道館公演をもって活動を終了したが、' \
           'その後2020年に「再生」と称して解散時のメンバーで再始動することが発表された。所属レコード会社はユニバーサルミュージック。' \
           'シンガーソングライターの椎名林檎を中心に、2003年に結成された5人組のロックバンド。2012年2月29日の日本武道館公演をもって' \
           '活動を終了。その後、2020年に「再生」と称して解散時のメンバーで再始動することが発表された。'
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
        phraser = PhraseConstructor('en')
        phrase, token = phraser.tokenize_and_stem_and_phrase(test_en)
        LOGGER.info(token)
        for k, v in phrase.items():
            assert 'NOUN' in v['pos'] or 'PROPN' in v['pos'], 'noun not found {}'.format(v['pos'])
        LOGGER.info(phrase.keys())

        phraser = PhraseConstructor('ja')
        phrase, token = phraser.tokenize_and_stem_and_phrase(test_ja)
        LOGGER.info(token)
        for k, v in phrase.items():
            assert 'NOUN' in v['pos'] or 'PROPN' in v['pos'], 'noun not found {}'.format(v['pos'])

    def test_tokenize_and_stem(self):
        phraser = PhraseConstructor('en')
        token = phraser.tokenize_and_stem(test_en)
        LOGGER.info(token)

        phraser = PhraseConstructor('ja')
        token = phraser.tokenize_and_stem(test_ja)
        LOGGER.info(token)


if __name__ == "__main__":
    unittest.main()
