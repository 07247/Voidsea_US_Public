import pandas as pd


def filter_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    只保留美股正常交易时间：
    09:30 <= 时间 < 16:00
    """

    data = df.copy()

    # Alpaca 返回的数据通常是：
    # MultiIndex -> symbol, timestamp
    if isinstance(data.index, pd.MultiIndex):
        timestamps = data.index.get_level_values("timestamp")
    else:
        timestamps = data.index

    # 确保时间有时区
    if timestamps.tz is None:
        raise ValueError("timestamp 没有时区信息，无法判断美股交易时间")

    # 转换成纽约时间
    ny_time = timestamps.tz_convert("America/New_York")

    # 转换为当天的分钟数
    minutes = ny_time.hour * 60 + ny_time.minute

    # 09:30 = 570 分钟
    # 16:00 = 960 分钟
    mask = (minutes >= 570) & (minutes < 960)

    return data[mask]


def add_sma(
    df: pd.DataFrame, short_window: int = 20, long_window: int = 50
) -> pd.DataFrame:
    """
    添加 SMA 短期均线和长期均线
    """

    data = df.copy()

    short_name = f"sma_{short_window}"
    long_name = f"sma_{long_window}"

    # 如果是 Alpaca 的 MultiIndex 数据
    if isinstance(data.index, pd.MultiIndex) and "symbol" in data.index.names:
        data[short_name] = data.groupby(level="symbol")["close"].transform(
            lambda x: x.rolling(window=short_window, min_periods=short_window).mean()
        )

        data[long_name] = data.groupby(level="symbol")["close"].transform(
            lambda x: x.rolling(window=long_window, min_periods=long_window).mean()
        )

    else:
        data[short_name] = (
            data["close"].rolling(window=short_window, min_periods=short_window).mean()
        )

        data[long_name] = (
            data["close"].rolling(window=long_window, min_periods=long_window).mean()
        )

    return data


def add_sma_signal(
    df: pd.DataFrame, short_window: int = 20, long_window: int = 50
) -> pd.DataFrame:
    """
    SMA 金叉/死叉信号：

    BUY:
        SMA20 从下向上穿过 SMA50

    SELL:
        SMA20 从上向下穿过 SMA50

    其他:
        HOLD
    """

    data = df.copy()

    short_name = f"sma_{short_window}"
    long_name = f"sma_{long_window}"

    short_ma = data[short_name]
    long_ma = data[long_name]

    # 多股票情况下，每只股票分别计算上一根 K 线
    if isinstance(data.index, pd.MultiIndex) and "symbol" in data.index.names:
        previous_short = short_ma.groupby(level="symbol").shift(1)

        previous_long = long_ma.groupby(level="symbol").shift(1)

    else:
        previous_short = short_ma.shift(1)
        previous_long = long_ma.shift(1)

    # 默认 HOLD
    data["signal"] = "HOLD"

    # 金叉
    buy_condition = (short_ma > long_ma) & (previous_short <= previous_long)

    # 死叉
    sell_condition = (short_ma < long_ma) & (previous_short >= previous_long)

    data.loc[buy_condition, "signal"] = "BUY"
    data.loc[sell_condition, "signal"] = "SELL"

    return data


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算 RSI 指标

    RSI > 70：通常认为偏超买
    RSI < 30：通常认为偏超卖
    """

    data = df.copy()

    rsi_name = f"rsi_{period}"

    def calculate_rsi(close):
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))

        return rsi

    # 多股票
    if isinstance(data.index, pd.MultiIndex) and "symbol" in data.index.names:
        data[rsi_name] = data.groupby(level="symbol")["close"].transform(calculate_rsi)

    # 单股票
    else:
        data[rsi_name] = calculate_rsi(data["close"])

    return data


def add_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    计算 MACD

    fast=12   快速 EMA
    slow=26   慢速 EMA
    signal=9  信号线 EMA
    """

    data = df.copy()

    close = data["close"]

    # 多股票数据
    if isinstance(data.index, pd.MultiIndex) and "symbol" in data.index.names:
        ema_fast = close.groupby(level="symbol").transform(
            lambda x: x.ewm(span=fast, adjust=False).mean()
        )

        ema_slow = close.groupby(level="symbol").transform(
            lambda x: x.ewm(span=slow, adjust=False).mean()
        )

        macd = ema_fast - ema_slow

        macd_signal = macd.groupby(level="symbol").transform(
            lambda x: x.ewm(span=signal, adjust=False).mean()
        )

    # 单股票数据
    else:
        ema_fast = close.ewm(span=fast, adjust=False).mean()

        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow

        macd_signal = macd.ewm(span=signal, adjust=False).mean()

    # MACD 主线
    data["macd"] = macd

    # Signal 信号线
    data["macd_signal"] = macd_signal

    # 柱状图
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    return data


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算 ATR（Average True Range）
    """

    data = df.copy()

    atr_name = f"atr_{period}"

    # ==========================
    # 多股票 / Alpaca MultiIndex
    # ==========================
    if isinstance(data.index, pd.MultiIndex) and "symbol" in data.index.names:
        # 每只股票分别取得上一根K线收盘价
        previous_close = data.groupby(level="symbol")["close"].shift(1)

        # True Range 三种情况
        tr1 = data["high"] - data["low"]

        tr2 = (data["high"] - previous_close).abs()

        tr3 = (data["low"] - previous_close).abs()

        # 每根K线取最大值
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 每只股票分别计算 ATR
        data[atr_name] = true_range.groupby(level="symbol").transform(
            lambda x: x.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        )

    # ==========================
    # 普通单股票 DataFrame
    # ==========================
    else:
        previous_close = data["close"].shift(1)

        tr1 = data["high"] - data["low"]

        tr2 = (data["high"] - previous_close).abs()

        tr3 = (data["low"] - previous_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        data[atr_name] = true_range.ewm(
            alpha=1 / period, adjust=False, min_periods=period
        ).mean()

    return data
