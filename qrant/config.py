"""全局配置：A股交易成本、默认标的池。"""
from __future__ import annotations

from dataclasses import dataclass, field

TRADING_DAYS = 252

# 默认自选：覆盖 A股大盘/中小盘/成长、黄金、国债、海外，便于分散
DEFAULT_WATCHLIST: dict[str, str] = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "518880": "黄金ETF",
    "511010": "国债ETF",
    "513100": "纳指ETF",
}

# 用于“市场状态”判断的基准指数
BENCHMARK_INDEX = "sh000300"
BENCHMARK_NAME = "沪深300指数"


def is_etf(code: str) -> bool:
    """按代码前缀判断是否为场内基金（ETF/LOF），ETF 免印花税。"""
    c = str(code).strip().lower().lstrip("shz.")[-6:]
    return c[:2] in {"51", "52", "56", "58", "15", "16", "50"}


# 非A股权益类：债券、黄金/商品、跨境(QDII) —— 不应跟随沪深300的牛熊过滤
_NON_EQUITY_PREFIX = ("511", "518", "513", "159934", "159937", "159941", "159920", "159980", "159985",
                      "159981", "159501", "159509", "159632", "159659", "159866")


def is_ashare_equity(code: str) -> bool:
    c = str(code).strip().lower()[-6:]
    return not c.startswith(_NON_EQUITY_PREFIX)


_CHINEXT_ETF = {"159915", "159952", "159948", "159908", "159977", "159949", "159967", "159966"}


def price_limit(code: str) -> float:
    """涨跌停幅度：创业板/科创板 20%，北交所 30%，其余 10%。"""
    c = str(code).strip().lower()[-6:]
    if c.startswith(("300", "301", "688", "689", "588")) or c in _CHINEXT_ETF:
        return 0.20  # 含科创板ETF(588)与创业板ETF
    if c.startswith(("8", "4", "92")):
        return 0.30
    return 0.10


@dataclass
class CostModel:
    """A股交易成本模型（2026 现行规则）。"""

    commission_rate: float = 0.00025  # 佣金 万2.5
    min_commission: float = 5.0  # 单笔最低 5 元
    stamp_duty: float = 0.0005  # 印花税 0.05%，仅卖出；ETF 免征
    transfer_fee: float = 0.00001  # 过户费 0.001%，双向
    slippage: float = 0.0005  # 滑点（单边，按价格比例）
    lot_size: int = 100  # 一手 100 股
    etf: bool = True  # 是否为 ETF（决定是否收印花税）
    limit_pct: float = 0.10  # 涨跌停幅度

    @classmethod
    def for_symbol(cls, code: str, **kw) -> "CostModel":
        return cls(etf=is_etf(code), limit_pct=price_limit(code), **kw)

    def fee(self, value: float, side: str) -> float:
        """一笔成交的总费用（不含滑点，滑点体现在成交价里）。"""
        if value <= 0:
            return 0.0
        f = max(self.min_commission, value * self.commission_rate)
        f += value * self.transfer_fee
        if side == "sell" and not self.etf:
            f += value * self.stamp_duty
        return f


@dataclass
class RiskConfig:
    """风控层参数。每一项都可以在界面中开关，以便观察它的贡献与代价。"""

    vol_target_on: bool = True
    target_vol: float = 0.15  # 年化目标波动率
    vol_lookback_short: int = 20
    vol_lookback_long: int = 60  # 取短期/长期波动率较大者，偏保守
    max_exposure: float = 1.0  # 不加杠杆

    regime_on: bool = True
    regime_ma: int = 200  # 基准在 200 日均线之下视为熊市
    bear_exposure: float = 0.0  # 熊市时仓位上限（0 = 空仓）；0.3 样本外未稳定胜出，见 validate_oos.py

    atr_stop_on: bool = False  # 默认关：消融实验显示其在宽基ETF上增加打脸交易；个股建议开启
    atr_period: int = 20
    atr_mult: float = 3.0

    breaker_on: bool = False
    breaker_dd: float = 0.15  # 策略自身回撤超过 15% 触发熔断
    breaker_cooldown: int = 20  # 冷却交易日
    breaker_resume_ma: int = 60  # 冷却结束且价格站上该均线才恢复

    rebalance_band: float = 0.10  # 仓位偏离 >10% 才调仓，抑制无谓换手
    cash_rate: float = 0.012  # 空仓资金年化收益（货币基金/国债逆回购），所有对比线一视同仁
    decide_freq: str = "W"  # 决策频率：D 每日 / W 每周最后一个交易日（降低换手）

    extra: dict = field(default_factory=dict)
