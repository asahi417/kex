""" UnitTest for preprocessor"""

import unittest
import keyphraser

PROCESSOR_EN = keyphraser.algorithms.processing.Processing(language='en')
STR_CLEANER_EN = keyphraser.algorithms.processing.CleaningStr(language='en')
PROCESSOR_JP = keyphraser.algorithms.processing.Processing(language='jp')
STR_CLEANER_JP = keyphraser.algorithms.processing.CleaningStr(language='jp')


class TestPreprocessing(unittest.TestCase):
    """Test"""

    def test_regx_manual_filter(self):
        """regx filter check"""
        target = ["""   ** Exact Sciences Corp &lt;EXAS.O&gt; shares up 11.5 pct to $73.22 after co's preliminary
        Q4 and FY 2018 sales beat expectations     ** EXAS expects Q4 revenue of $142.5-$143.5 mln vs Street consensus
        of $129.8 mln, according to data from IBES Refinitiv     ** Says Cologuard Q4 test volumes increase 66 pct 
        yr/yr to ~292,000 [nPn8nwqHTa]     ** BTIG analyst Sean Lavin says Cologuard tests beat his brokerage's
        Street-high estimate of 269,000; adds preliminary results position co well for driving upside to current
        consensus revenue estimates for FY 2019     ** Wall Street forecasts EXAS to generate revenue of $700.9 mln
        in 2019 (Refinitiv data)     ** Lavin, who rates EXAS a "buy" with $95 PT, awarded 5 of 5 stars by Refinitiv
        for both estimate accuracy and recommendation performance on the stock     ** On Aug 22, EXAS soared &gt;30 
        pct after co inked deal with Pfizer &lt;PFE.N&gt; to promote Cologuard, its non-invasive stool screening test
        for detecting colorectal cancer [nL3N1VD4DA]     ** EXAS now up 16 pct so far this year, following gains of 20
        pct in 2018 and nearly 300 pct in 2017     ** EXAS scheduled to present on Tues at the annual JP Morgan
        healthcare conference in San Francisco, the granddaddy of all conferences for the sector     ** M&amp;A 
        chatter is in the air as the four-day event, which features &gt;450 public and private co investor 
        presentations, expects &gt;9,000 attendees [nL1N1Z418] [nL3N1Z74DE]   
        ((lance.tupper.tr.com@reuters.net lance.tupper@tr.com 646-223-5017))"""]
        expect = [[
            'Exact Sciences Corp', 'FY', 'sales', 'expectations', 'EXAS', 'Q4 revenue', 'Street consensus',
            'IBES Refinitiv', 'Cologuard Q4 test volumes', 'BTIG analyst Sean Lavin', 'Cologuard tests', 'brokerage',
            'Street-high estimate', 'preliminary results position', 'consensus revenue estimates', 'Wall Street',
            'revenue', 'Refinitiv data', 'Lavin', 'buy', 'PT', 'stars', 'Refinitiv', 'estimate accuracy',
            'recommendation performance', 'stock', 'co inked deal', 'Pfizer', 'Cologuard',
            'non-invasive stool screening test', 'colorectal cancer', 'gains', 'annual JP Morgan',
            'healthcare conference', 'San Francisco', 'granddaddy', 'conferences', 'sector', 'M&A', 'chatter', 'air',
            'four-day event', 'private co investor', 'presentations', 'attendees']]

        for t, e in zip(target, expect):
            out = PROCESSOR_EN(t)
            raw_token = [out[k]['raw'][0] for k in out.keys()]
            assert set(raw_token) == set(e)

    def test_phraser(self):
        """ check phrasing"""

        target = ['New York is a central city of United States in Sept.',
                  'New York is in September (or Sept. or Sep.) a central city of United States.',
                  'New York is a central city of United States, U.S.']
        expected = [['New York', 'central city', 'United States'],
                    ['New York', 'central city', 'United States'],
                    ['New York', 'central city', 'United States', 'U.S.']]

        for t, g in zip(target, expected):
            out = PROCESSOR_EN(t)
            raw_token = [out[k]['raw'][0] for k in out.keys()]
            assert raw_token == g

    def test_empty(self):
        """ empty input should be ignore"""
        target = ['', ' ', '.']
        expect = [{}, {}, {}]
        for t, e in zip(target, expect):
            processed = PROCESSOR_EN(t)
            assert processed == e
            processed = PROCESSOR_JP(t)
            print(processed, e)
            assert processed == e

    def test_numeric_term(self):
        """ numeric term should be excluded from candidate list"""
        out = PROCESSOR_EN('The stock price went up 5% in U.S. market.')
        assert ['stock price', 'u.s. market'] == list(out.keys())
        out = PROCESSOR_JP(
            '4月の今日は死ぬほど暑いね。100度あるんじゃない？１００度だよ。 '
            'https://github.com/cogentlabs/keyphraser/blob/master/keyphraser/algorithms/processing/en.py これが github の url で　asdfsdf@gmail.com'
        )
        assert ['今日', 'GitHub', 'URL'] == list(out.keys())

    def test_halfspacing(self):
        """ redundant halfspace should be removed """
        target = '''ANALYST RECOMMENDATIONS Of the 14 analysts who disclosed their recommendation on the SSAB stock, 10 answered Positive, three Neutral and one Negative.      - The following brokerages and investment banks participated in the poll:       Firm                  Outlook  Morgan Stanley        Overweight  Bank of America       Buy  Merrill Lynch           Handelsbanken         Buy  Capital Markets         Citi Investment       Buy  Research                Deutsche Bank         Hold  JP Morgan             Overweight  Macquarie Securities  Overweight  UBS                   Sell  DNB Markets           Hold  Jefferies             Buy  International           Kepler Cheuvreux      Buy  Exane BNP Paribas     Overweight  Goldman Sachs &amp; Co    Buy  Pareto Securities     Hold        Estimates were collected from 2018-09-25
to  2018-10-19      Data provided by Inquiry Financial Europe AB (www.consensusestimates.com)   (Gdynia Newsroom)  ((gdynia.newsroom@thomsonreuters.com; +48 58 772 0920;))'''
        expect = '''ANALYST RECOMMENDATIONS Of the YYY analysts who disclosed their recommendation on the SSAB stock, YYY answered Positive, three Neutral and one Negative. - The following brokerages and investment banks participated in the poll: . Firm . Outlook . Morgan Stanley . Overweight . Bank of America . Buy . Merrill Lynch . Handelsbanken . Buy . Capital Markets . Citi Investment . Buy . Research . Deutsche Bank . Hold . JP Morgan . Overweight . Macquarie Securities . Overweight . UBS . Sell . DNB Markets . Hold . Jefferies . Buy . International . Kepler Cheuvreux . Buy . Exane BNP Paribas . Overweight . Goldman Sachs & Co . Buy . Pareto Securities . Hold . Estimates were collected from YYY to . YYY . Data provided by Inquiry Financial Europe AB XXX . (Gdynia Newsroom) . ZZZ YYY YYY YYY YYY'''
        out = STR_CLEANER_EN.process(target)
        assert out == expect


if __name__ == "__main__":
    unittest.main()
