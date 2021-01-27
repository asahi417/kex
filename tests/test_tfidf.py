""" UnitTest for TFIDF/ExpandRank """
import unittest
import logging

from kex import TFIDF, ExpandRank

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
TEST_DOCS = [
"""
Efficient discovery of grid services is essential for the success of
grid computing. The standardization of grids based on web
services has resulted in the need for scalable web service
discovery mechanisms to be deployed in grids Even though UDDI
has been the de facto industry standard for web-services
discovery, imposed requirements of tight-replication among
registries and lack of autonomous control has severely hindered
its widespread deployment and usage. With the advent of grid
computing the scalability issue of UDDI will become a roadblock
that will prevent its deployment in grids. In this paper we present
our distributed web-service discovery architecture, called DUDE
(Distributed UDDI Deployment Engine). DUDE leverages DHT
(Distributed Hash Tables) as a rendezvous mechanism between
multiple UDDI registries. DUDE enables consumers to query
multiple registries, still at the same time allowing organizations to
have autonomous control over their registries.. Based on
preliminary prototype on PlanetLab, we believe that DUDE
architecture can support effective distribution of UDDI registries
thereby making UDDI more robust and also addressing its scaling
issues. Furthermore, The DUDE architecture for scalable
distribution can be applied beyond UDDI to any Grid Service
Discovery mechanism.
""".replace('\n', ' '),
"""
Grids are inherently heterogeneous and dynamic. One important
problem in grid computing is resource selection, that is, finding
an appropriate resource set for the application. Another problem
is adaptation to the changing characteristics of the grid 
environment. Existing solutions to these two problems require that a 
performance model for an application is known. However, 
constructing such models is a complex task. In this paper, we investigate
an approach that does not require performance models. We start an
application on any set of resources. During the application run, we
periodically collect the statistics about the application run and 
deduce application requirements from these statistics. Then, we adjust
the resource set to better fit the application needs. This approach 
allows us to avoid performance bottlenecks, such as overloaded WAN
links or very slow processors, and therefore can yield significant
performance improvements. We evaluate our approach in a number
of scenarios typical for the Grid.
""".replace('\n', ' '),
"""
Best effort packet-switched networks, like the Internet, do
not offer a reliable transmission of packets to applications
with real-time constraints such as voice. Thus, the loss of
packets impairs the application-level utility. For voice this
utility impairment is twofold: on one hand, even short bursts
of lost packets may decrease significantly the ability of the
receiver to conceal the packet loss and the speech signal 
playout is interrupted. On the other hand, some packets may
be particular sensitive to loss as they carry more important
information in terms of user perception than other packets.
We first develop an end-to-end model based on loss 
runlengths with which we can describe the loss distribution
within a flow. These packet-level metrics are then linked to
user-level objective speech quality metrics. Using this 
framework, we find that for low-compressing sample-based codecs
(PCM) with loss concealment isolated packet losses can be
concealed well, whereas burst losses have a higher perceptual
impact. For high-compressing frame-based codecs (G.729)
on one hand the impact of loss is amplified through error
propagation caused by the decoder filter memories, though
on the other hand such coding schemes help to perform loss
concealment by extrapolation of decoder state. Contrary to
sample-based codecs we show that the concealment 
performance may break at transitions within the speech signal
however.
We then propose mechanisms which differentiate between
packets within a voice data flow to minimize the impact of
packet loss. We designate these methods as intra-flow loss
recovery and control. At the end-to-end level, identification
of packets sensitive to loss (sender) as well as loss 
concealment (receiver) takes place. Hop-by-hop support schemes
then allow to (statistically) trade the loss of one packet,
which is considered more important, against another one of
the same flow which is of lower importance. As both 
packets require the same cost in terms of network transmission,
a gain in user perception is obtainable. We show that 
significant speech quality improvements can be achieved and
additional data and delay overhead can be avoided while
still maintaining a network service which is virtually 
identical to best effort in the long term.
""".replace('\n', ' ')]


class TestTFIDF(unittest.TestCase):
    """Test tfidf"""

    def test_tfidf(self):
        model = TFIDF()
        model.train(TEST_DOCS, export_directory='./cache/unittest/priors')
        test = 'Efficient discovery of grid services is essential for the success of grid computing. The ' \
               'standardization of grids based on web services has resulted in the need for scalable web service ' \
               'discovery mechanisms to be deployed in grids Even though UDDI has been the de facto industry standard ' \
               'for web-services discovery, imposed requirements of tight-replication among registries and lack of ' \
               'autonomous control has severely hindered its widespread deployment and usage. With the advent of grid ' \
               'computing the scalability issue of UDDI will become a roadblock that will prevent its deployment in ' \
               'grids. In this paper we present our distributed web-service discovery architecture, called DUDE ' \
               '(Distributed UDDI Deployment Engine). DUDE leverages DHT (Distributed Hash Tables) as a rendezvous ' \
               'mechanism between multiple UDDI registries. DUDE enables consumers to query multiple registries, ' \
               'still at the same time allowing organizations to have autonomous control over their registries. ' \
               'Based on preliminary prototype on PlanetLab, we believe that DUDE architecture can support effective ' \
               'distribution of UDDI registries thereby making UDDI more robust and also addressing its scaling ' \
               'issues. Furthermore, The DUDE architecture for scalable distribution can be applied beyond UDDI to ' \
               'any Grid Service Discovery mechanism.'
        logging.info(test)
        out = model.get_keywords(test, 3)
        for i in out:
            logging.info(i)

        # load
        model = TFIDF()
        model.load('./cache/test_tfidf')
        test = 'Efficient discovery of grid services is essential for the success of grid computing. The ' \
               'standardization of grids based on web services has resulted in the need for scalable web service ' \
               'discovery mechanisms to be deployed in grids Even though UDDI has been the de facto industry standard ' \
               'for web-services discovery, imposed requirements of tight-replication among registries and lack of ' \
               'autonomous control has severely hindered its widespread deployment and usage. With the advent of grid ' \
               'computing the scalability issue of UDDI will become a roadblock that will prevent its deployment in ' \
               'grids. In this paper we present our distributed web-service discovery architecture, called DUDE ' \
               '(Distributed UDDI Deployment Engine). DUDE leverages DHT (Distributed Hash Tables) as a rendezvous ' \
               'mechanism between multiple UDDI registries. DUDE enables consumers to query multiple registries, ' \
               'still at the same time allowing organizations to have autonomous control over their registries. ' \
               'Based on preliminary prototype on PlanetLab, we believe that DUDE architecture can support effective ' \
               'distribution of UDDI registries thereby making UDDI more robust and also addressing its scaling ' \
               'issues. Furthermore, The DUDE architecture for scalable distribution can be applied beyond UDDI to ' \
               'any Grid Service Discovery mechanism.'
        logging.info(test)
        out = model.get_keywords(test, 3)
        for i in out:
            logging.info(i)

        dist = model.distribution_word(test)
        logging.info(dist)

    def test_expand_rank(self):
        model = ExpandRank()
        model.train(TEST_DOCS, export_directory='./cache/unittest/priors')
        test = 'Efficient discovery of grid services is essential for the success of grid computing. The ' \
               'standardization of grids based on web services has resulted in the need for scalable web service ' \
               'discovery mechanisms to be deployed in grids Even though UDDI has been the de facto industry standard ' \
               'for web-services discovery, imposed requirements of tight-replication among registries and lack of ' \
               'autonomous control has severely hindered its widespread deployment and usage. With the advent of grid ' \
               'computing the scalability issue of UDDI will become a roadblock that will prevent its deployment in ' \
               'grids. In this paper we present our distributed web-service discovery architecture, called DUDE ' \
               '(Distributed UDDI Deployment Engine). DUDE leverages DHT (Distributed Hash Tables) as a rendezvous ' \
               'mechanism between multiple UDDI registries. DUDE enables consumers to query multiple registries, ' \
               'still at the same time allowing organizations to have autonomous control over their registries. ' \
               'Based on preliminary prototype on PlanetLab, we believe that DUDE architecture can support effective ' \
               'distribution of UDDI registries thereby making UDDI more robust and also addressing its scaling ' \
               'issues. Furthermore, The DUDE architecture for scalable distribution can be applied beyond UDDI to ' \
               'any Grid Service Discovery mechanism.'
        logging.info(test)
        out = model.get_keywords(test, 3)
        for i in out:
            logging.info(i)

        # load
        model = ExpandRank()
        model.load('./cache/unittest/priors')
        test = 'Efficient discovery of grid services is essential for the success of grid computing. The ' \
               'standardization of grids based on web services has resulted in the need for scalable web service ' \
               'discovery mechanisms to be deployed in grids Even though UDDI has been the de facto industry standard ' \
               'for web-services discovery, imposed requirements of tight-replication among registries and lack of ' \
               'autonomous control has severely hindered its widespread deployment and usage. With the advent of grid ' \
               'computing the scalability issue of UDDI will become a roadblock that will prevent its deployment in ' \
               'grids. In this paper we present our distributed web-service discovery architecture, called DUDE ' \
               '(Distributed UDDI Deployment Engine). DUDE leverages DHT (Distributed Hash Tables) as a rendezvous ' \
               'mechanism between multiple UDDI registries. DUDE enables consumers to query multiple registries, ' \
               'still at the same time allowing organizations to have autonomous control over their registries. ' \
               'Based on preliminary prototype on PlanetLab, we believe that DUDE architecture can support effective ' \
               'distribution of UDDI registries thereby making UDDI more robust and also addressing its scaling ' \
               'issues. Furthermore, The DUDE architecture for scalable distribution can be applied beyond UDDI to ' \
               'any Grid Service Discovery mechanism.'
        logging.info(test)
        out = model.get_keywords(test, 3)
        for i in out:
            logging.info(i)


if __name__ == "__main__":
    unittest.main()
