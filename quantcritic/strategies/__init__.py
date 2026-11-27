"""Built-in crypto-perpetuals trading strategies and registry."""

from quantcritic.strategies.base import Strategy
from quantcritic.strategies.bollinger_squeeze import BollingerSqueezeStrategy
from quantcritic.strategies.builder import BuilderStrategy, StrategySpec
from quantcritic.strategies.donchian_breakout import DonchianBreakoutStrategy
from quantcritic.strategies.ema_trend import EmaTrendStrategy
from quantcritic.strategies.funding_contrarian import FundingContrarianStrategy
from quantcritic.strategies.leaky_demo import LeakyDemoStrategy
from quantcritic.strategies.registry import get_strategy, list_strategies, reload_custom_strategies
from quantcritic.strategies.rsi_reversion import RsiReversionStrategy

__all__ = [
    "BollingerSqueezeStrategy",
    "BuilderStrategy",
    "DonchianBreakoutStrategy",
    "EmaTrendStrategy",
    "FundingContrarianStrategy",
    "LeakyDemoStrategy",
    "RsiReversionStrategy",
    "Strategy",
    "StrategySpec",
    "get_strategy",
    "list_strategies",
    "reload_custom_strategies",
]
