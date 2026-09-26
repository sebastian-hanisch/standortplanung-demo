"""Der Zufallsgenerator ist aus den Vorgängern kopiert (SplitMix64, Portfolio-Standard): derselbe Strom wie in ssp-demo und netzwerksimplex-demo."""

import ufl_scenario as sc


def test_splitmix64_stream_is_the_portfolio_standard():
    rng = sc.SplitMix64(1)
    assert [rng.next() for _ in range(2)] == [10451216379200822465, 13757245211066428519]
    assert [sc.SplitMix64(7).below(101) for _ in range(3)] == [sc.SplitMix64(7).below(101)] * 3
