""" UnitTest for keyphrase extraction algorithm in general
 - English
 - verb
"""

import unittest
import keyphraser
from glob import glob
import json
import os

INSTANCES = dict(
    TopicRank=keyphraser.algorithm_instance('TopicRank'),
    TextRank=keyphraser.algorithm_instance('TextRank')
)
INSTANCES_VERB = dict(
    TopicRank=keyphraser.algorithm_instance('TopicRank', add_verb=True),
    TextRank=keyphraser.algorithm_instance('TextRank', add_verb=True)
)

TEST_SENT_NO_KW = 'This is a sample'
TEST_SENT = 'I live in Tokyo and London at the same time'
TEST_SENT_EXPECT = ['Tokyo', 'London']


class TestAlgorithm(unittest.TestCase):
    """Test basic algorithm output"""

    @staticmethod
    def get_result(output, key='raw'):
        def tmp(__output):
            if len(__output) == 0:
                return __output
            else:
                return [o[key][0] for o in __output]

        return [tmp(_o) for _o in output]

    def do_test(self, target, expected, algorithm=None, add_verb=False, n=5):
        if algorithm is None:
            if add_verb:
                instances = list(INSTANCES_VERB.values())
            else:
                instances = list(INSTANCES.values())
        else:
            if add_verb:
                instances = [INSTANCES_VERB[algorithm]]
            else:
                instances = [INSTANCES[algorithm]]
        for t, e in zip(target, expected):
            for instance in instances:
                out = instance.extract(t, n)
                out = self.get_result(out, 'raw')
                # if not out == e:
                #     print(out, e)
                for a, b in zip(out, e):
                    if not set(a) == set(b):
                        print()
                        print(' - output:', set(a))
                        print(' - expect:', set(b))
                        assert set(a) == set(b)

    def test_numeric_input(self):
        """ numeric input should be ignore and return empty list"""
        target = [[0.1, 100], [0.1], [110], [500, TEST_SENT], [90, TEST_SENT_NO_KW]]
        expected = [[[], []], [[]], [[]], [[], TEST_SENT_EXPECT], [[], []]]
        self.do_test(target, expected, add_verb=False)
        # self.do_test(target, expected, add_verb=True)

    def test_empty(self):
        """empty input should be ignore and return empty list"""
        target = [[''], [], ['', TEST_SENT], ['', TEST_SENT_NO_KW], [None]]
        expected = [[[]], [], [[], TEST_SENT_EXPECT], [[], []], [[]]]
        self.do_test(target, expected, add_verb=False)
        # self.do_test(target, expected, add_verb=True)

    def test_verb(self):
        """keyphrase algorithm requires at least 2 phrase, so return nothing for single token"""
        target = [['This is a test to retrieve verb so should contain lots of verb but only one noun like Tokyo']]
        expected = [[['Tokyo', 'retrieve', 'contain', 'lots', 'test', 'noun', 'verb']]]
        self.do_test(target, expected, algorithm='TopicRank', add_verb=True, n=100)

    def test_small_token(self):
        """keyphrase algorithm requires at least 2 phrase, so return nothing for single token"""
        target = [['London'], ['Tokyo', TEST_SENT]]
        expected = [[[]], [[], TEST_SENT_EXPECT]]
        self.do_test(target, expected, add_verb=False)
        # self.do_test(target, expected, add_verb=True)

    def test_basic(self):
        """basic test"""
        target = [
            [
                "China's coal imports jumped 12.2 percent in April from a month earlier, data showed on Monday, even afterAustralian shipments were disrupted by a powerful cyclone and Beijing banned shipments of the fuel from North Korea.     Data from the General Administration of Customs of China showed shipments into the world's largest buyer of the fuel hit 24.78 million tonnes, up 32 percent from the same period last year.     The increase reflects a sustained pick-up in buying by utilities and steel mills of cheaper foreign coal amid a domestic coal price rally triggered by Beijing's efforts to phase out overcapacity. [nB9N1HZ01W]     It also suggests buyers found alternative supplies to Australia after a category four Cyclone Debbie hit the major coal-mining state of Queensland in late March, disrupting mines and shutting down most of the rail transport system. [nL4N1I51II] [nL4N1HZ26N]      In March, Russia ramped up sales to China after Beijing prohibited high-quality anthracite imports from North Korea that are typically used for steelmaking, in compliance with United Nations' sanctions against Pyongyang's nuclear and missile programme. [nL4N1HY3N7] [nL4N1I51II]          The figures include lignite, a type of coal with lower heating value that is largely supplied by Indonesia.      For more details, click on [TRADE/CN]   (Reporting by Josephine Mason and Lusha Zhang; Editing by Richard Pullin)  ((Josephine.Mason@thomsonreuters.com; +86 10 66271210; Reuters Messaging: josephine.mason.reuters.com@reuters.net))  Keywords: CHINA ECONOMY/TRADE COAL",
                "A knife-fielding man suspected of mental illness killed two people and injured 18 in China, the official Xinhua news agency reported.     Violent crime is rare in China compared with many other countries, but there has been a series of knife and axe attacks in recent years, many targeting children.      Twenty people were taken to hospital after the Sunday attack in Guizhou province in the southwest and two died, Xinhua said.     It did not give any details of the victims.      The attacker, aged 30, is being held in police custody and the case was under investigation.      Xinhua cited the suspect's father as saying his son had a history of mental illness.     In January, a man wounded 11 children with a blade at their kindergarten in the Guangxi region. Seven children were wounded in a November attack by a man with a knife outside in another area. [nL4N1EU35N][nL4N1DQ2R2]   (Reporting by Brenda Goh; Editing by Robert Birsel)  ((brenda.goh@thomsonreuters.com; +86)(0)(21 6104 1763; Reuters Messaging: brenda.goh.thomsonreuters.com@reuters.net))  Keywords: CHINA ATTACK/",
                "Guangzhou Rural Commercial Bank Co Ltd (GRCB) launched a Hong Kong initial public offering worth as much as $1.1 billion on Monday, seeking funds for potential M&A and to open new branches as it expands its lending and investment businesses.         The IPO for China's fifth-largest rural commercial bank by assets consists of 1.58 billion shares offered in an indicative range of HK$4.99-HK$5.27 each, according to a term sheet of the deal seen by Reuters.     That would be equivalent to around 16.5 percent of the lender after the offering, valuing it at as much as $6.7 billion.     GRCB did not immediately reply to a Reuters request for comment.     The lender secured commitments worth about $431 million from three investors, including $195.1 million each from a unit of HNA Group and from Aeon Life Insurance Company Ltd, which is controlled by billionaire Wang Jianlin's Dalian Wanda Group.     Investment firm International Merchants Holdings plans to buy $40 million worth of shares.     The IPO is set to be priced on June 13, with its debut on the Hong Kong stock exchange slated for June 20.     ABC International, CCB International, China International Capital Corp Ltd (CICC) and China Merchants Securities were hired as sponsors for the IPO, GRCB said in its preliminary IPO prospectus.         ($1 = 7.7902 Hong Kong dollars)   (Reporting by Julie Zhu and Fiona Lau of IFR; Writing by Elzio Barreto; Editing by Edwina Gibbs)  ((elzio.barreto@thomsonreuters.com;)(852)(2843-1608; Reuters Messaging: elzio.barreto.thomsonreuters.com@reuters.net))  Keywords: GRCBANK IPO/"
                "The Chinese yuan and Indian rupee are expected to shed some of this year's gains and weaken slightly against the dollar over the coming 12 months if the U.S. Federal Reserve raises interest rates further as expected, a Reuters poll showed.     China's yuan <CNY=CFXS> hit its highest in just over half a year on Wednesday and was last trading around 6.79 against the dollar.[CNY/]     The currency has gained nearly 2 percent so far this year, with half of that coming just in the last month.      The move comes on the heels of faster-than-expected growth of 6.9 percent in the Chinese economy in the first quarter of this year. But that was largely reliant on fiscal stimulus - the country's total social financing reached a record 6.93 trillion yuan ($137 billion) for the same period.     Another reason for concern is Moody's Investors' Service decision to cut China's credit rating for the first time in nearly 30 years, with debt continuing to rise. [nL4N1IQ07P]      The yuan is forecast to weaken to 7.05 per dollar in 12 months, according to the poll of over 50 foreign exchange analysts taken this week, even as market confusion reigns over China's plans to tweak the currency's midpoint calculation for a second time this year.     China said on Friday it was introducing an unspecified counter-cyclical factor, intended to discourage speculation and persistent depreciation pressure, though the currency had been largely stable earlier in the year as the dollar floundered.          The latest poll predictions were similar to last month's and based on similar Fed rate expectations for two more rate incrases this year.     While the Fed is widely expected to raise rates in June, which would be broadly supportive for the dollar, the currency has been whipsawed in recent weeks as hopes about the U.S. administration's economic growth plans have faded.     The top-down drivers for EM (emerging market) currencies suggest slightly stronger performance in spot for the near-term and depreciation over a 12-month horizon, noted Dirk Willer, head of emerging market strategy at Citi.      This implies only moderate strengthening in the USD, contributing to stability in the CNY and, by extension, other emerging market currencies. This is assumed to take place against the backdrop of moderately higher U.S. rates.          RUPEE SEEN WEAKENING MORE THAN 2 PERCENT     Separately, the Indian rupee <INR=IN> is forecast to weaken to 66.00 per dollar over the next year, a more than 2 percent fall from where it was trading on Wednesday after gaining more than 5 percent so far this year.     Official data on Wednesday showed India's economic growth unexpectedly slowed to its weakest in more than two years in the first three months of 2017, stripping the country of its status as the world's fastest growing major economy.     Still, the Reserve Bank of India is expected to stand pat when it meets on June 7, but carrying a less hawkish tone according to Reuters poll published this week. [RBI/INT]                  (For other stories from the FX poll: [nL3N1IX3SD])  ($1 = 6.7881 Chinese yuan renminbi)   (Polling by Shaloo Shrivastava and Khushboo Mittal; Editing by Ross Finley and Kim Coghill)  ((krishna.eluri@thomsonreuters.com; +91 80 67496065; Reuters Messaging: krishna.eluri.thomsonreuters.com@reuters.net))  Keywords: FOREX POLL/ASIA",
                "Coal giant Shenhua Group Corp Ltd [SHGRP.UL] and top-five state power producer China Guodian Corp [CNGUO.UL] are in talks to merge some assets, sources told Reuters on Monday, as part of a broader shake-up of China's debt-ridden state-owned sector.     Several of the power firms' listed units suspended trading in their shares on Monday citing a planned significant event, fanning market speculation over a merger. The shares of other listed units that continued trading rose sharply.     The talks come as the government seeks to streamline state-owned enterprises (SOEs), including those in the energy sector, by creating huge, globally competitive conglomerates. It has merged 15 SOEs since 2015 and currently manages 103 - a number that could eventually fall to about 40, state media reported.     In the latest merger talks, a person with direct knowledge of the matter said China's largest coal miner, Shenhua Group, would take over Guodian unit GD Power Development Co Ltd <600795.SS>.     After merging Guodian's GD Power into Shenhua, Shenhua will consider acquiring coal-fired power assets from the remaining top power firms, said the person, who was not authorised to speak with media on the matter and so declined to be identified.     A second person with knowledge of the matter said merger talks were at a preliminary stage, and that the option of completely merging the two parents was likely to be tabled later.     Shenhua Group, its listed unit China Shenhua Energy Co Ltd <601088.SS>, China Guodian and GD Power could not be immediately reached for comment.     We think this merger is very likely to occur, NSBO Research said in a note to clients, referring to a merger of some form between Shenhua Group and China Guodian. Any merger would be positive for their listed units, NSBO Research said.          OVER-CAPACITY     China Guodian is one of five state power producers formed in 2002 after the restructuring of China's state-owned power sector monopoly, along with China Huadian Corp [CNHUA.UL], State Power Investment Corp [CPWRI.UL], China Huaneng Group [HUANP.UL] and China Datang Corp [SASADT.UL].     In March, the chairman of China's State-owned Assets Supervision and Administration Commission (SASAC) said SOE restructuring would focus on the steel, coal, heavy equipment and coal-fired power sectors this year to tackle over-capacity. [nL3N1GM2MA]     With smog-plagued China moving toward cleaner fuel, a merger of Shenhua Group with a major state power provider such as China Guodian - also a leading hydropower and renewables developer - could ease its dependence on coal.     Other listed units of China Guodian - Guodian Changyuan Electric Power Co Ltd <000966.SZ>, Yantai LongYuan Power Technology Co Ltd <300105.SZ> and Ningxia Younglight Chemicals Co Ltd <000635.SZ> - also suspended share trading on Monday.     The listed firms said in statements they had been informed by their parent companies about a significant event that involved major uncertainties and required regulatory approvals.     Trading continued in the Hong Kong-listed shares of China Shenhua Energy <1088.HK> as well as Guodian Technology & Environment Group Corp Ltd <1296.HK>. The prices of both rose sharply in morning trading. [nL3N1J21GU]   (Reporting by Meng Meng, Adam Jourdan and David Stanway; Editing by Stephen Coates and Christopher Cushing)  ((adam.jourdan@thomsonreuters.com; +86 21 6104 1778; Reuters Messaging: adam.jourdan.thomsonreuters.com@reuters.net))  Keywords: CHINA POWER/SHENHUA",
            ],
            ["China's coal imports jumped 12.2 percent in April from a month earlier, data showed on Monday, even afterAustralian shipments were disrupted by a powerful cyclone and Beijing banned shipments of the fuel from North Korea.     Data from the General Administration of Customs of China showed shipments into the world's largest buyer of the fuel hit 24.78 million tonnes, up 32 percent from the same period last year.     The increase reflects a sustained pick-up in buying by utilities and steel mills of cheaper foreign coal amid a domestic coal price rally triggered by Beijing's efforts to phase out overcapacity. [nB9N1HZ01W]     It also suggests buyers found alternative supplies to Australia after a category four Cyclone Debbie hit the major coal-mining state of Queensland in late March, disrupting mines and shutting down most of the rail transport system. [nL4N1I51II] [nL4N1HZ26N]      In March, Russia ramped up sales to China after Beijing prohibited high-quality anthracite imports from North Korea that are typically used for steelmaking, in compliance with United Nations' sanctions against Pyongyang's nuclear and missile programme. [nL4N1HY3N7] [nL4N1I51II]          The figures include lignite, a type of coal with lower heating value that is largely supplied by Indonesia.      For more details, click on [TRADE/CN]   (Reporting by Josephine Mason and Lusha Zhang; Editing by Richard Pullin)  ((Josephine.Mason@thomsonreuters.com; +86 10 66271210; Reuters Messaging: josephine.mason.reuters.com@reuters.net))  Keywords: CHINA ECONOMY/TRADE COAL"],
            ["A knife-fielding man suspected of mental illness killed two people and injured 18 in China, the official Xinhua news agency reported.     Violent crime is rare in China compared with many other countries, but there has been a series of knife and axe attacks in recent years, many targeting children.      Twenty people were taken to hospital after the Sunday attack in Guizhou province in the southwest and two died, Xinhua said.     It did not give any details of the victims.      The attacker, aged 30, is being held in police custody and the case was under investigation.      Xinhua cited the suspect's father as saying his son had a history of mental illness.     In January, a man wounded 11 children with a blade at their kindergarten in the Guangxi region. Seven children were wounded in a November attack by a man with a knife outside in another area. [nL4N1EU35N][nL4N1DQ2R2]   (Reporting by Brenda Goh; Editing by Robert Birsel)  ((brenda.goh@thomsonreuters.com; +86)(0)(21 6104 1763; Reuters Messaging: brenda.goh.thomsonreuters.com@reuters.net))  Keywords: CHINA ATTACK/",]
        ]
        print()
        print('*** with verb ***')
        for t in target:
            for instance in INSTANCES.values():
                out = instance.extract(t, 5)
                for _o in out:
                    assert len(_o) == 5
                    print(instance.__module__, [k['raw'][0] for k in _o])
        print('*** with verb ***')
        for t in target:
            for instance in INSTANCES_VERB.values():
                out = instance.extract(t, 5)
                for _o in out:
                    assert len(_o) == 5
                    print(instance.__module__, [k['raw'][0] for k in _o])
        print()

    def test_manual_exceptions(self):
        """picked up sample raises error"""
        # 1st: very long phrase
        # 2nd: 'electric pickup truck maker Rivian -sources Amazon.com Inc AMZN.O'
        # 3rd: '--------+---------------'
        target = [
            ["""    * SSAB third-quarter results     * Due on Thursday, Oct 26     * Q3 adj. oper profit seen at SEK 1.66 billion      Oct 23 - Following is a table of forecasts for Swedish steel maker SSAB &lt;SSABa.ST&gt; third-quarter results, according to a poll of analysts.           All figures in millions of SEK, except EPS and Dividend per share which are in SEK.            2018:Q3                                           Yr   Chan                         Mean  Media   High    Low  No    Ago    ge%  Prev.                            n                                      Q  Sales          18,85  18,83  19,66  17,62  14  16,18  16.49  19,26                     8      2      1      0          8             3   - SSAB        4,683  4,784  4,954  3,990  12  3,627  29.11  5,142  Special                                                        Steels                                                          - SSAB        8,190  8,166  8,750  7,696  12  7,245  13.04  8,892  Europe                                                          - Americas    4,484  4,483  4,766  3,727  12  3,340  34.25  4,040   - Tibnor      1,924  1,915  2,100  1,733  12  1,733  11.02  2,253   - Ruukki      1,668  1,673  1,731  1,591  12  1,640   1.71  1,578  Construction                                                    - Other       -2,08  -2,15  -1,44  -2,66  12  -1,39  -49.1    n/a                     3      3      3      8          7      1    EBITDA         2,583  2,526  3,142  2,347  14  2,016  28.13  2,582  Adjusted                                                       Operating      1,656  1,616  2,181  1,463  14  1,089  52.07  1,630  Profit -                                                       Adjusted                                                        - SSAB          420    434    499    345  11    353  18.98    522  Special                                                        Steels                                                          - SSAB          672    635    904    576  11  1,031  -34.8    907  Europe                                                    2     - Americas      707    691  1,130    364  11    468  51.07    365   - Tibnor       49.9   47.6   71.5   34.7  11   65.0  -23.2   83.0                                                            2     - Ruukki       94.2   93.0    124   68.5  11    137  -31.2   59.0  Construction                                              4    Non-recurring  -7.35   0.00   0.00   -103  14   0.00    n/a   0.00  items                                                          Operating      1,648  1,616  2,181  1,404  14  1,089  51.33  1,630  profit                                                         Pretax Profit  1,447  1,426  1,985  1,129  14    864  67.48  1,427  Net Profit     1,140  1,121  1,580    844  14    580  96.55  1,313  EPS             1.11   1.09   1.53   0.82  14   0.56  98.21   1.27  2018                                              Yr   Chan                  Mean  Media   High    Low  No    Ago    ge%                            n                             Sales          74,42  74,72  76,15  70,13  14  66,05  12.66                     1      4      8      6          9     - SSAB        19,18  19,50  19,89  16,82  12  16,05  19.49  Special            2      1      6      2          3    Steels                                                   - SSAB        33,49  33,39  34,47  32,66  12  31,04   7.87  Europe             2      4      6      3          8     - Americas    16,20  16,42  16,74  14,34  12  12,72  27.32                     4      7      7      9          7     - Tibnor      8,315  8,235  8,696  8,009  12  7,821   6.32   - Ruukki      5,860  5,848  6,058  5,684  12  5,773   1.51  Construction                                             - Other       -8,67  -8,82  -7,48  -9,82  12  -7,36  -17.8                     4      1      8      5          3      1  EBITDA         9,318  9,228  10,12  8,720  14  7,591  22.75  Adjusted                         8                      Operating      5,571  5,588  6,284  5,070  14  3,839  45.12  Profit -                                                Adjusted                                                 - SSAB        1,662  1,698  1,915  1,287  11  1,465  13.45  Special                                                 Steels                                                   - SSAB        3,100  2,949  4,223  2,695  11  2,988   3.75  Europe                                                   - Americas    1,651  1,713  2,199    680  11    183  802.1                                                            9   - Tibnor        244    244    268    225  11    252  -3.17   - Ruukki        136    132    217   98.5  11    171  -20.4  Construction                                              7  Non-recurring  -7.35   0.00   0.00   -103  14   0.00    n/a  items                                                   Operating      5,564  5,588  6,284  4,967  14  3,839  44.93  profit                                                  Pretax Profit  4,814  4,850  5,497  4,068  14  2,863  68.15  Net Profit     3,965  4,099  4,367  3,361  14  2,295  72.77  EPS             3.85   3.98   4.24   3.26  14   2.23  72.65  Dividend per    1.59   1.64   2.03   1.00  13   1.00     59  share                                                   2019                                                           Mean  Media   High    Low  No                            n                  Sales          74,89  75,62  80,81  65,21  14                     0      0      6      4     - SSAB        19,59  20,15  22,13  15,91  12  Special            3      4      2      4    Steels                                        - SSAB        33,14  33,65  35,82  28,83  12  Europe             8      5      8      1     - Americas    16,58  16,54  19,05  13,95  12                     4      7      7      5     - Tibnor      8,200  8,265  8,689  7,391  12   - Ruukki      5,949  5,965  6,120  5,663  12  Construction                                  - Other       -8,52  -8,73  -6,00  -9,82  12                     6      6      0      5    EBITDA         9,583  9,575  12,13  6,862  14  Adjusted                         3           Operating      5,958  5,799  8,458  3,860  14  Profit -                                     Adjusted                                      - SSAB        1,767  1,948  2,129    972  11  Special                                      Steels                                        - SSAB        2,939  3,031  3,868  1,918  11  Europe                                        - Americas    1,875  1,755  3,243  1,125  11   - Tibnor        207    203    282    129  11   - Ruukki        171    161    268    101  11  Construction                                 Non-recurring   0.00   0.00   0.00   0.00  14  items                                        Operating      5,958  5,799  8,458  3,860  14  profit                                       Pretax Profit  5,341  5,223  8,051  3,034  14  Net Profit     4,235  4,191  6,396  2,328  14  EPS             4.11   4.07   6.21   2.26  14  Dividend per    1.83   1.73   2.76   1.20  13  share                                        2020                                                           Mean  Media   High    Low  No                            n                  Sales          73,23  72,96  79,84  64,98  13                     1      9      7      7    EBITDA         9,261  9,189  12,47  7,144  13  Adjusted                         0           Operating      5,937  6,146  9,142  3,484  13  Profit -                                     Adjusted                                     Operating      5,937  6,146  9,142  3,484  13  profit                                       Pretax Profit  5,499  5,545  9,175  2,930  13  Net Profit     4,345  4,429  7,289  2,183  13  EPS             4.22   4.30   7.08   2.12  13  Dividend per    1.82   1.76   2.88   1.07  13  share                                              ANALYST RECOMMENDATIONS Of the 14 analysts who disclosed their recommendation on the SSAB stock, 10 answered Positive, three Neutral and one Negative.      - The following brokerages and investment banks participated in the poll:       Firm                  Outlook  Morgan Stanley        Overweight  Bank of America       Buy  Merrill Lynch           Handelsbanken         Buy  Capital Markets         Citi Investment       Buy  Research                Deutsche Bank         Hold  JP Morgan             Overweight  Macquarie Securities  Overweight  UBS                   Sell  DNB Markets           Hold  Jefferies             Buy  International           Kepler Cheuvreux      Buy  Exane BNP Paribas     Overweight  Goldman Sachs &amp; Co    Buy  Pareto Securities     Hold        Estimates were collected from 2018-09-25 to  2018-10-19      Data provided by Inquiry Financial Europe AB (www.consensusestimates.com)   (Gdynia Newsroom)  ((gdynia.newsroom@thomsonreuters.com; +48 58 772 0920;))"""],
            ["""[nL1N2070KZ]     • Amazon, GM in talks to invest in electric pickup truck maker Rivian -sources Amazon.com Inc AMZN.O and General Motors Co GM.N are in talks to invest in Rivian Automotive LLC in a deal that would value the U.S. electric pickup truck manufacturer at between $1 billion and $2 billion, people familiar with the matter told Reuters on Tuesday. """],
            ["""STATUS     LAST_REVISED  SCHEDULED OUTAGES (OUTAGE REQUEST RECEIVED BY PJM DISPATCHING PERSONNEL)        OPEN/CLOSED---| (. . . . outage type . . . )  ZONE/CO  FACILITY_NAME                                     START_DATE TIME    END_DATE  TIME  | (. . . . c a u s e s . . . )  +--------+------------------------------------------------+-----------------+-----------------+-+---------+-----------------+  APSS     BRKR FTMARTIN 500 KV  FTMARTIN FL8 CB       CB   10-NOV-2018 2330  11-NOV-2018 0230  O  Active   11/10/2018 23:47            1 hr.       Approved   |                                                                                                  (Continuous                )                                                                                                  (Maintenance: Gas (SF6)                            )                                                                                                  (Operational: Emergency                            )                                                                                                  (10-NOV-2018 2330   11-NOV-2018 0230    11/10/2018 22:54)                                                                                                  (Active       11/10/2018 23:47)                                                                                                  (Approved     11/10/2018 23:47)  +--------+------------------------------------------------+-----------------+-----------------+-+---------+-----------------+  PL       CAP  JUNIATA  500 KV  JUNIATA  500-2    CAP      10-NOV-2018 2342  12-NOV-2018 1600  O  Active """]
        ]
        expected = [[['SSAB', 'oper profit', 'Americas', 'investment banks', 'Swedish steel maker SSAB']],
                    [['electric pickup truck maker Rivian', 'General Motors Co GM.N', 'Amazon.com Inc AMZN.O', 'talks', 'Rivian Automotive LLC']],
                    [['KV', 'OUTAGES', 'FTMARTIN FL8 CB', 'JUNIATA', 'CAP']]]
        self.do_test(target, expected, algorithm='TopicRank', add_verb=False)


if __name__ == "__main__":
    unittest.main()