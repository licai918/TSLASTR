# %% [markdown]
# # 特斯拉反转策略回测与优化 (TSLA Reversal Strategy Backtest & Optimization)
#
# 本 Notebook 使用 Python 对基于 Pine Script 的特斯拉股票反转交易策略进行回测和参数优化。
# 主要利用 `vectorbt` 进行高效的回测，`optuna` 进行超参数优化，以及 `yfinance` 获取股票数据。

# %% [markdown]
# ## 1. 导入库 (Import Libraries)
# 导入所有必需的 Python 库。

# %%
import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
import optuna
import plotly.graph_objects as go
from numba import njit
import pandas_ta as ta # 使用 pandas_ta 来简化指标计算
import warnings
from joblib import parallel_backend
import os
import time
from tqdm import tqdm
from datetime import timedelta
import sys

# 记录总执行时间
total_script_start_time = time.time()
print(f"脚本开始执行时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_script_start_time))}")

# 在脚本结尾添加总执行时间显示函数
def print_total_execution_time():
    total_script_end_time = time.time()
    total_duration = total_script_end_time - total_script_start_time
    print("\n" + "="*50)
    print(f"脚本总执行时间: {str(timedelta(seconds=int(total_duration)))}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_script_start_time))}")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_script_end_time))}")
    print("="*50)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='vectorbt')
optuna.logging.set_verbosity(optuna.logging.WARNING) # 减少 optuna 的日志输出

# 获取CPU核心数
cpu_count = os.cpu_count()
print(f"系统CPU核心数: {cpu_count}")

# %% [markdown]
# ## 2. 配置参数 (Configuration)
# 设置回测和优化的关键参数，如股票代码、时间范围和初始资金。

# %%
# --- 回测和数据参数 ---
TICKER = "TSLA"
START_DATE = "2024-12-17" # 修改为更合理的历史日期范围
END_DATE = "2025-04-04"   # 修改为当前或过去的日期
INTERVAL = "1h"          # 数据频率 ('1d', '1h', '30m', etc.) - 从'2h'改为'1h'，因为YF不支持2h间隔

# --- 策略默认参数 (来自 Pine Script) ---
# 这些参数将在优化时被 Optuna 覆盖，但在这里设置一个默认值用于初始回测
initial_params = {
    'rsi_length': 7,
    'iv_length': 126,
    'macd_fast': 8,
    'macd_slow': 17,
    'macd_signal': 9,
    'adx_length': 14,  # 保留ADX计算，但不使用权重
    'volume_ma_period': 20,
    'volume_spike_window': 20,  # 成交量峰值窗口
    'volume_spike_mult': 2.0,   # 成交量峰值倍数
    'rsi_low_percentile': 30,     # 从20改为30，更容易触发低RSI信号
    'rsi_high_percentile': 70,    # 从80改为70，更容易触发高RSI信号
    'iv_low_percentile': 30,      # 从20改为30，更容易触发低IV信号
    'iv_high_percentile': 70,     # 从80改为70，更容易触发高IV信号
    'take_profit_mult': 2.0,
    'stop_loss_mult': 2.0,
    'bull_score_threshold': 0.1,     # 多头差额阈值，允许负值
    'bear_score_threshold': -0.1,    # 空头差额阈值，允许负值
    # --- 移除SMA和ADX的权重，保留其他权重参数 ---
    'rsi_bull_weight': 0.35,       # 增加权重以弥补移除的SMA和ADX
    'iv_bull_weight': 0.25,        # 增加权重以弥补移除的SMA和ADX
    'macd_bull_weight': 0.3,       # 增加权重以弥补移除的SMA和ADX
    'volume_bull_weight': 0.1,
    'rsi_bear_weight': 0.35,       # 增加权重以弥补移除的SMA和ADX
    'iv_bear_weight': 0.25,        # 增加权重以弥补移除的SMA和ADX
    'macd_bear_weight': 0.3,       # 增加权重以弥补移除的SMA和ADX
    'volume_bear_weight': 0.1,
}

# --- vectorbt 回测设置 ---
INITIAL_CAPITAL = 100000
PYRAMIDING = 1 # 允许的金字塔加仓次数
PCT_OF_EQUITY = 100 # 每次交易使用的资金比例

# --- Optuna 优化设置 ---
N_TRIALS = 15000 # 增加优化尝试次数，从30次增加到500次进行更彻底的参数搜索
OPTIMIZATION_METRIC = 'Total Return [%]' # 用于优化的目标指标 ('Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]')

# %% [markdown]
# ## 3. 数据获取与预处理 (Data Fetching and Preprocessing)
# 使用 `yfinance` 下载指定时间范围内的股票数据，并进行基本的清洗。

# %%
# 下载数据
price_data = yf.download(TICKER, start=START_DATE, end=END_DATE, interval=INTERVAL)

# 检查数据是否为空
if price_data.empty:
    print(f"错误：无法下载 {TICKER} 的数据，请检查日期范围和时间间隔")
    print("尝试使用其他时间间隔，例如 '1d'(日线)、'1h'(小时线)、'15m'(15分钟线)")
    print("支持的时间间隔: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")
    print_total_execution_time()
    exit(1)  # 退出程序，返回错误代码

# 检查并处理列名 (处理可能的 MultiIndex)
if isinstance(price_data.columns, pd.MultiIndex):
    # 如果是 MultiIndex，取第一级的名称并转为小写
    price_data.columns = [col[0].lower() for col in price_data.columns]
else:
    # 如果是普通 Index，直接转为小写
    price_data.columns = price_data.columns.str.lower()

# 检查和处理缺失值 (例如，向前填充)
price_data.ffill(inplace=True) # 注意：更复杂的缺失值处理可能需要根据数据情况调整

print(f"数据已下载，从 {price_data.index.min()} 到 {price_data.index.max()}")
print(price_data.head())
print(price_data.info())


# %% [markdown]
# ## 4. 指标计算函数 (Indicator Calculation Functions)
# 定义函数来计算策略所需的各项技术指标。这里使用 `pandas-ta` 来简化标准指标的计算，并自定义隐含波动率的计算。

# %%
def calculate_iv_rank(close: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """
    计算模拟的隐含波动率 (IV Rank) 基于历史波动率 (HV)。
    注意：这是对 Pine Script 中 IV Rank 的模拟，并非真实的期权隐含波动率。
    """
    # 使用 pandas 内置的 rolling 标准差计算，避免 ta.stdev 可能返回 None 的问题
    # 计算滚动标准差作为历史波动率
    hv = close.rolling(window=period, min_periods=max(1, period//2)).std() / close * 100  # 百分比形式

    # 防止除以零或无效值
    hv.replace([np.inf, -np.inf], np.nan, inplace=True)
    hv.fillna(method='ffill', inplace=True) # 简单填充NaN

    # 计算 HV Rank (百分位数排名)
    min_hv = hv.rolling(window=period, min_periods=max(1, period//2)).min()
    max_hv = hv.rolling(window=period, min_periods=max(1, period//2)).max()
    # 防止除以零
    range_hv = max_hv - min_hv
    range_hv[range_hv == 0] = 1e-6 # 避免除以零

    hv_rank = (hv - min_hv) / range_hv * 100
    hv_rank.fillna(0, inplace=True) # 填充初始 NaN

    return hv, hv_rank

def calculate_indicators(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    计算所有策略所需的指标。
    移除SMA指标，只使用MACD和VOLUME SPIKE作为主要技术指标。
    """
    df = data.copy()
    
    # 如果数据为空，返回空DataFrame
    if df.empty:
        print("警告：数据为空，无法计算指标")
        return df

    # --- 标准指标 (使用 pandas-ta) ---
    df['rsi'] = ta.rsi(df['close'], length=params['rsi_length'])
    
    # 添加错误处理，确保MACD计算正确
    try:
        macd = ta.macd(df['close'], fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'])
        if macd is not None:
            df['macd_line'] = macd[f'MACD_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}']
            df['macd_signal'] = macd[f'MACDs_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}']
            df['macd_hist'] = macd[f'MACDh_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'] # MACD 柱状图
        else:
            print("警告：MACD计算返回None，使用替代方法计算MACD")
            # 手动计算MACD
            ema_fast = df['close'].ewm(span=params['macd_fast'], adjust=False).mean()
            ema_slow = df['close'].ewm(span=params['macd_slow'], adjust=False).mean()
            df['macd_line'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd_line'].ewm(span=params['macd_signal'], adjust=False).mean()
            df['macd_hist'] = df['macd_line'] - df['macd_signal']
    except Exception as e:
        print(f"MACD计算出错：{e}，使用替代方法计算")
        # 手动计算MACD作为备选
        ema_fast = df['close'].ewm(span=params['macd_fast'], adjust=False).mean()
        ema_slow = df['close'].ewm(span=params['macd_slow'], adjust=False).mean()
        df['macd_line'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd_line'].ewm(span=params['macd_signal'], adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['macd_signal']
        
    df['macd_change'] = df['macd_line'].diff() # MACD 线的变化

    # 修复: 使用正确的 adx 函数并添加错误处理
    try:
        adx = ta.adx(df['high'], df['low'], df['close'], length=params['adx_length'])
        if adx is not None:
            df['adx'] = adx[f'ADX_{params["adx_length"]}']
            df['di_plus'] = adx[f'DMP_{params["adx_length"]}']
            df['di_minus'] = adx[f'DMN_{params["adx_length"]}']
        else:
            print("警告：ADX计算返回None，使用替代方法计算ADX")
            # 可以添加备选计算方法，或者设置默认值
            df['adx'] = 25.0  # 设置一个默认值
            df['di_plus'] = 20.0
            df['di_minus'] = 20.0
    except Exception as e:
        print(f"ADX计算出错：{e}，使用默认值")
        df['adx'] = 25.0
        df['di_plus'] = 20.0
        df['di_minus'] = 20.0
    
    # 计算成交量均线 (必需，用于Volume Spike指标)
    df['volume_ma'] = ta.sma(df['volume'], length=params['volume_ma_period'])
    
    # Volume Spike指标 - 成交量突破 (作为主要技术指标保留)
    volume_spike_window = params.get('volume_spike_window', 20)  # 默认20个周期
    volume_spike_mult = params.get('volume_spike_mult', 2.0)     # 默认2倍于均线为峰值
    
    # 计算相对于均线的成交量倍数
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # 标记成交量峰值 - 当成交量超过均线的volume_spike_mult倍
    df['volume_spike'] = df['volume_ratio'] > volume_spike_mult
    
    # 计算成交量变化率 - 较前一周期的变化百分比
    df['volume_change'] = df['volume'].pct_change() * 100
    
    # 成交量强度指标 - 基于成交量相对均线的比率和变化率
    df['volume_strength'] = df['volume_ratio'] * (1 + df['volume_change']/100)
    
    # 填充NaN值
    df['volume_ratio'].fillna(1.0, inplace=True)
    df['volume_change'].fillna(0.0, inplace=True)
    df['volume_strength'].fillna(1.0, inplace=True)

    # --- 自定义 IV Rank ---
    df['hv'], df['iv_rank'] = calculate_iv_rank(df['close'], params['iv_length'])

    # --- 历史分位数阈值 (使用 rolling percentile) ---
    # 注意：vectorbt 通常在整个数据集上计算指标，然后应用策略。
    # Pine Script 的 percentile 计算可能略有不同（基于历史可用数据）。
    # 这里我们使用 rolling percentile 来模拟。窗口大小设为 250 作为近似。
    rolling_window = 250
    df['rsi_low_thresh'] = df['rsi'].rolling(window=rolling_window, min_periods=max(50, params['rsi_length'])).quantile(params['rsi_low_percentile'] / 100.0)
    df['rsi_high_thresh'] = df['rsi'].rolling(window=rolling_window, min_periods=max(50, params['rsi_length'])).quantile(params['rsi_high_percentile'] / 100.0)
    df['iv_low_thresh'] = df['iv_rank'].rolling(window=rolling_window, min_periods=max(50, params['iv_length'])).quantile(params['iv_low_percentile'] / 100.0)
    df['iv_high_thresh'] = df['iv_rank'].rolling(window=rolling_window, min_periods=max(50, params['iv_length'])).quantile(params['iv_high_percentile'] / 100.0)

    # 填充初始 NaN 值
    df.fillna(method='bfill', inplace=True) # 向后填充阈值，确保开始时有值
    df.fillna(method='ffill', inplace=True) # 向前填充剩余 NaN

    return df


# %% [markdown]
# ## 5. 策略信号生成函数 (Strategy Signal Generation Function)
# 定义函数根据计算出的指标和评分逻辑生成买入和卖出信号。

# %%
@njit
def calculate_scores_nb(
    close: np.ndarray,
    volume: np.ndarray,
    rsi: np.ndarray,
    iv_rank: np.ndarray,
    macd_line: np.ndarray,
    macd_signal: np.ndarray,
    macd_change: np.ndarray,
    adx: np.ndarray,
    volume_ratio: np.ndarray,
    volume_spike: np.ndarray,
    volume_strength: np.ndarray,
    volume_change: np.ndarray,
    rsi_low_thresh: np.ndarray,
    rsi_high_thresh: np.ndarray,
    iv_low_thresh: np.ndarray,
    iv_high_thresh: np.ndarray,
    rsi_bull_w: float, iv_bull_w: float, macd_bull_w: float, volume_bull_w: float,
    rsi_bear_w: float, iv_bear_w: float, macd_bear_w: float, volume_bear_w: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba JIT 编译的函数，用于快速计算多头和空头评分。
    简化版本: 移除SMA指标，只使用MACD和VOLUME SPIKE作为主要技术指标。
    ADX仅作为过滤器使用，但不计入权重得分。
    """
    n = len(close)
    bullish_scores = np.full(n, 0.0)
    bearish_scores = np.full(n, 0.0)

    for i in range(1, n): # 从 1 开始以访问前一天的 macd_change
        # 初始化评分为零
        bull_score = 0.0
        bear_score = 0.0
        
        # ADX过滤器 - 只有当ADX > 20时才计算评分
        if adx[i] > 20:
            # Volume Spike - 使用成交量强度和峰值
            volume_factor = 0.0
            if volume_spike[i] and volume_change[i] > 0:  # 成交量峰值且上升
                volume_factor = volume_bull_w * min(volume_strength[i], 3.0) / 3.0  # 限制最大值
            
            # Bullish score components
            # RSI低于低阈值，多头信号
            if rsi[i] <= rsi_low_thresh[i] * 1.1:  # 放宽10%
                bull_score += rsi_bull_w
                
            # IV高于高阈值，多头信号 (波动率高可能意味着拐点)
            if iv_rank[i] >= iv_high_thresh[i] * 0.9:  # 放宽10%
                bull_score += iv_bull_w
                
            # MACD线在信号线上方，且MACD线上升，多头信号
            if macd_line[i] > macd_signal[i] * 0.95 or macd_change[i] > 0:  # 放宽条件，只需满足一个
                bull_score += macd_bull_w
                
            # 添加成交量评分
            bull_score += volume_factor
            
            # =========== 空头信号 ===========
            # RSI高于高阈值，空头信号
            if rsi[i] >= rsi_high_thresh[i] * 0.9:  # 放宽10%
                bear_score += rsi_bear_w
                
            # IV低于低阈值，空头信号
            if iv_rank[i] <= iv_low_thresh[i] * 1.1:  # 放宽10%
                bear_score += iv_bear_w
                
            # MACD线在信号线下方，且MACD线下降，空头信号
            if macd_line[i] < macd_signal[i] * 1.05 or macd_change[i] < 0:  # 放宽条件，只需满足一个
                bear_score += macd_bear_w
                
            # 添加成交量评分 - 空头可能也需要高成交量
            if volume_spike[i] and volume_change[i] < 0:  # 成交量峰值且下降
                bear_score += volume_bear_w * min(volume_strength[i], 3.0) / 3.0  # 限制最大值
        
        # 保存评分
        bullish_scores[i] = bull_score
        bearish_scores[i] = bear_score

    return bullish_scores, bearish_scores


def generate_signals(indicator_df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    根据评分生成入场信号，包括多头和空头信号。
    使用双阈值差额评分机制生成交易信号，与更新后的Pine Script策略保持一致。
    返回多头入场、多头离场、空头入场、空头离场信号。
    """
    # 确保所有数组都转换为numpy的float64类型，以避免Numba类型错误
    close_arr = indicator_df['close'].values.astype(np.float64)
    volume_arr = indicator_df['volume'].values.astype(np.float64)
    rsi_arr = indicator_df['rsi'].values.astype(np.float64)
    iv_rank_arr = indicator_df['iv_rank'].values.astype(np.float64)
    macd_line_arr = indicator_df['macd_line'].values.astype(np.float64)
    macd_signal_arr = indicator_df['macd_signal'].values.astype(np.float64)
    macd_change_arr = indicator_df['macd_change'].values.astype(np.float64)
    adx_arr = indicator_df['adx'].values.astype(np.float64)
    
    # 移除SMA相关数组
    # 仍然保留volume_ma_arr，因为Volume Spike需要它
    volume_ma_arr = indicator_df['volume_ma'].values.astype(np.float64)
    
    # 新增Volume Spike相关数组
    volume_ratio_arr = indicator_df['volume_ratio'].values.astype(np.float64)
    volume_spike_arr = indicator_df['volume_spike'].values.astype(np.float64)  # 将布尔值转为float64
    volume_strength_arr = indicator_df['volume_strength'].values.astype(np.float64)
    volume_change_arr = indicator_df['volume_change'].values.astype(np.float64)
    
    rsi_low_thresh_arr = indicator_df['rsi_low_thresh'].values.astype(np.float64)
    rsi_high_thresh_arr = indicator_df['rsi_high_thresh'].values.astype(np.float64)
    iv_low_thresh_arr = indicator_df['iv_low_thresh'].values.astype(np.float64)
    iv_high_thresh_arr = indicator_df['iv_high_thresh'].values.astype(np.float64)

    # 使用转换后的数组调用Numba函数，移除SMA和ADX权重参数
    bullish_scores, bearish_scores = calculate_scores_nb(
        close_arr,
        volume_arr,
        rsi_arr,
        iv_rank_arr,
        macd_line_arr,
        macd_signal_arr,
        macd_change_arr,
        adx_arr,
        volume_ratio_arr,
        volume_spike_arr,
        volume_strength_arr,
        volume_change_arr,
        rsi_low_thresh_arr,
        rsi_high_thresh_arr,
        iv_low_thresh_arr,
        iv_high_thresh_arr,
        params['rsi_bull_weight'], params['iv_bull_weight'], params['macd_bull_weight'], params['volume_bull_weight'],
        params['rsi_bear_weight'], params['iv_bear_weight'], params['macd_bear_weight'], params['volume_bear_weight']
    )

    indicator_df['bullish_score'] = bullish_scores
    indicator_df['bearish_score'] = bearish_scores
    
    # 计算评分差额
    indicator_df['score_difference'] = indicator_df['bullish_score'] - indicator_df['bearish_score']

    # 生成交易信号 - 基于双阈值差额评分机制
    # 多头信号: 评分差额 >= 多头差额阈值
    long_entries = (indicator_df['score_difference'] >= params['bull_score_threshold'])
    
    # 空头信号: 评分差额 <= 空头差额阈值
    short_entries = (indicator_df['score_difference'] <= params['bear_score_threshold'])
    
    # 正确实现Pine Script策略的退出逻辑
    # 在Pine Script中，退出信号是通过止盈止损设置的，而不是通过相反的信号
    long_exits = pd.Series(False, index=indicator_df.index)  # 初始化为全False
    short_exits = pd.Series(False, index=indicator_df.index)  # 初始化为全False
    
    return long_entries, long_exits, short_entries, short_exits


# %% [markdown]
# ## 6. 止盈止损计算函数 (Stop Loss / Take Profit Calculation)
# 实现基于历史波动率 (HV) 的动态止盈止损逻辑。

# %%
@njit
def hv_sl_tp_nb(close, hv, entry_idx, entry_price, sl_mult, tp_mult, is_long):
    """
    Numba JIT 编译函数，用于计算基于 HV 的止损和止盈价格。
    注意: vectorbt 的 exit 函数需要一个布尔序列，指示何时触发退出。
    直接计算价格比较困难，通常是通过 vbt.Portfolio 的 SL/TP 功能。
    这里我们先计算价格，稍后在 Portfolio 中使用。
    """
    sl_price = np.nan
    tp_price = np.nan

    # 获取触发入场那一刻的 HV 值
    current_hv = hv[entry_idx]
    if np.isnan(current_hv) or current_hv <= 0: # 使用一个默认值或前一个有效值
         # 查找 entry_idx 之前的最后一个有效 HV
        for k in range(entry_idx - 1, -1, -1):
            if not np.isnan(hv[k]) and hv[k] > 0:
                current_hv = hv[k]
                break
        if np.isnan(current_hv) or current_hv <= 0:
             current_hv = 1.0 # Fallback HV in percentage

    hv_decimal = current_hv / 100.0 # 转换为小数

    if is_long:
        tp_price = entry_price * (1 + hv_decimal * tp_mult)
        sl_price = entry_price * (1 - hv_decimal * sl_mult)
    else: # is_short
        tp_price = entry_price * (1 - hv_decimal * tp_mult)
        sl_price = entry_price * (1 + hv_decimal * sl_mult)

    return sl_price, tp_price

# 注意：vectorbt 的 from_signals 通常不直接处理这种动态 SL/TP。
# 它有内置的 sl_stop, tp_stop 参数，但它们通常是固定百分比或价格点。
# 实现 Pine Script 中的逐笔动态 SL/TP 需要更复杂的事件驱动回测或自定义信号。

# 替代方案：使用 vectorbt Portfolio 的内置功能，传递 HV * multiplier 作为止损/止盈的 *百分比*
# 这不完全等同于 Pine Script（它是基于入场价格计算绝对价格），但是一个可行的近似。

def get_sl_tp_signals(indicator_df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    """
    生成基于HV的动态止损/止盈百分比信号，用于vectorbt Portfolio。
    与Pine Script中的止盈止损逻辑一致。
    """
    # 确保输入数据是正确的类型，防止Numba错误
    close_arr = indicator_df['close'].values.astype(np.float64)
    hv_arr = indicator_df['hv'].values.astype(np.float64)
    
    # SL / TP 百分比 = HV (%) * multiplier / 100
    sl_stop_pct = indicator_df['hv'] * params['stop_loss_mult'] / 100.0
    tp_stop_pct = indicator_df['hv'] * params['take_profit_mult'] / 100.0

    # 确保百分比是正数且合理 (例如，限制最大值)
    sl_stop_pct = sl_stop_pct.clip(lower=0.001, upper=0.5) # 限制在 0.1% 到 50% 之间
    tp_stop_pct = tp_stop_pct.clip(lower=0.001, upper=0.5)

    # vectorbt会自动处理多头和空头的止盈止损方向
    # 对于多头：止损价格 = 入场价格 * (1 - sl_pct)，止盈价格 = 入场价格 * (1 + tp_pct)
    # 对于空头：止损价格 = 入场价格 * (1 + sl_pct)，止盈价格 = 入场价格 * (1 - tp_pct)
    # 这与Pine Script中的止盈止损逻辑完全一致
    
    return sl_stop_pct, tp_stop_pct


# %% [markdown]
# ## 7. 单次回测执行函数 (Single Backtest Execution Function)
# 定义一个函数来执行单次回测，方便在优化过程中调用。

# %%
def run_backtest(price_data: pd.DataFrame, params: dict, initial_capital=INITIAL_CAPITAL, pyramiding=PYRAMIDING, pct_equity=PCT_OF_EQUITY, verbose=False) -> vbt.Portfolio:
    """
    执行单次回测，支持同时进行多头和空头交易，完全匹配Pine Script策略逻辑。
    
    参数:
        verbose: 是否显示详细的过程信息
    """
    if verbose:
        print("开始执行回测流程...")
    try:
        # 1. 计算指标
        if verbose:
            print("正在计算技术指标...")
        indicator_df = calculate_indicators(price_data, params)
        
        if indicator_df.empty:
            print("警告：指标计算结果为空，无法继续回测")
            return None, None

        # 2. 生成入场/离场信号
        if verbose:
            print("正在生成交易信号...")
        long_entries, long_exits, short_entries, short_exits = generate_signals(indicator_df, params)

        # 3. 获取动态 SL/TP 百分比信号
        if verbose:
            print("正在计算止盈止损水平...")
        sl_stop_pct, tp_stop_pct = get_sl_tp_signals(indicator_df, params)

        # 4. 执行 vectorbt 回测
        if verbose:
            print("正在执行vectorbt回测...")
        portfolio = vbt.Portfolio.from_signals(
            close=indicator_df['close'],
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            sl_stop=sl_stop_pct,       # 每个时间点的止损百分比
            tp_stop=tp_stop_pct,       # 每个时间点的止盈百分比
            sl_trail=False,            # 是否使用追踪止损 (Pine Script 中没有)
            init_cash=initial_capital,
            size=pct_equity / 100.0,   # 基于权益的百分比下单
            size_type='percent',       # 百分比模式
            max_size=None,             # 每次信号最大比例，可选
            accumulate=pyramiding > 0, # 是否允许加仓 (对应 pyramiding)
            max_orders=pyramiding * 1000,  # 大幅增加允许的最大订单数，防止索引超出范围错误
            freq=INTERVAL              # K线频率
        )
        if verbose:
            print("回测执行完成！")
        return portfolio, indicator_df # 返回 portfolio 和包含指标的 df
        
    except Exception as e:
        print(f"回测执行过程中发生错误：{e}")
        import traceback
        traceback.print_exc()
        return None, None

# %% [markdown]
# ## 8. 执行初始回测 (Run Initial Backtest)
# 使用默认参数运行一次回测，查看基线性能。

# %%
# 输出 vectorbt 版本
print(f"vectorbt 版本: {vbt.__version__}")

# 检查 Portfolio.plot 支持的子图类型
try:
    # 创建一个小的测试数据集，测试 Portfolio.plot 的可用子图
    test_price = pd.Series(np.random.randn(100).cumsum() + 100, index=pd.date_range('2020-01-01', periods=100))
    test_entries = pd.Series(np.random.choice([True, False], size=100, p=[0.05, 0.95]), index=test_price.index)
    test_exits = pd.Series(np.random.choice([True, False], size=100, p=[0.05, 0.95]), index=test_price.index)
    
    test_pf = vbt.Portfolio.from_signals(test_price, test_entries, test_exits, init_cash=10000)
    
    # 通过尝试访问 plot 方法的 `_subplots` 属性或文档字符串来获取可用子图信息
    if hasattr(test_pf.plot, '__doc__') and test_pf.plot.__doc__:
        print("Portfolio.plot 文档:")
        print(test_pf.plot.__doc__)
    
    # 尝试获取plot方法的所有参数
    import inspect
    plot_signature = inspect.signature(test_pf.plot)
    print("Portfolio.plot 参数:")
    for param_name, param in plot_signature.parameters.items():
        print(f"  {param_name}: {param.default}")
        
    print("\n将继续尝试使用以下子图类型进行可视化:")
    print("cum_returns, orders, trade_pnl")
except Exception as e:
    print(f"诊断失败: {e}")

print("--- 开始初始回测 (使用默认参数) ---")
initial_backtest_start = time.time()
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(initial_backtest_start))}")

# 执行初始回测 - 设置verbose=True以显示详细信息
initial_portfolio, initial_indicator_df = run_backtest(price_data, initial_params, verbose=True)

# 检查回测是否成功
if initial_portfolio is None or initial_indicator_df is None:
    print("初始回测失败，无法继续执行优化过程。")
    print_total_execution_time()
    exit(1)

initial_backtest_end = time.time()
initial_backtest_duration = initial_backtest_end - initial_backtest_start
print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(initial_backtest_end))}")
print(f"回测用时: {str(timedelta(seconds=int(initial_backtest_duration)))}")

print("--- 初始回测结果 ---")
print(initial_portfolio.stats())

# 可视化初始回测结果
try:
    # 创建基本图表 - 不指定子图类型，让 vectorbt 默认决定
    fig_initial = initial_portfolio.plot()
    
    # 创建一个新的独立图表来显示评分
    fig_scores = go.Figure()
    
    fig_scores.add_trace(
        go.Scatter(
            x=initial_indicator_df.index, 
            y=initial_indicator_df['bullish_score'],
            name='多头评分', 
            line=dict(color='green')
        )
    )
    
    fig_scores.add_trace(
        go.Scatter(
            x=initial_indicator_df.index, 
            y=initial_indicator_df['bearish_score'],
            name='空头评分', 
            line=dict(color='red')
        )
    )
    
    # 添加评分差额线
    fig_scores.add_trace(
        go.Scatter(
            x=initial_indicator_df.index, 
            y=initial_indicator_df['score_difference'],
            name='评分差额', 
            line=dict(color='blue')
        )
    )
    
    # 添加阈值线
    fig_scores.add_hline(
        y=initial_params['bull_score_threshold'], 
        line=dict(color='green', dash='dash'), 
        name='多头差额阈值'
    )
    
    fig_scores.add_hline(
        y=initial_params['bear_score_threshold'], 
        line=dict(color='red', dash='dash'), 
        name='空头差额阈值'
    )
    
    # 添加零轴线
    fig_scores.add_hline(
        y=0, 
        line=dict(color='gray', dash='dot'), 
        name='零轴'
    )
    
    # 更新布局
    fig_scores.update_layout(
        title=f"{TICKER} 评分指标 ({START_DATE} to {END_DATE})",
        xaxis_title="日期",
        yaxis_title="评分",
        height=400
    )
    
    # 显示两个图表
    fig_initial.show()
    fig_scores.show()
except Exception as e:
    print(f"初始回测可视化错误: {e}")
    print("继续执行优化过程...")


# %% [markdown]
# ## 9. Optuna 优化目标函数 (Optuna Objective Function)
# 定义 `optuna` 用于优化的目标函数。该函数接收 `trial` 对象，建议参数，运行回测，并返回要优化的指标值。

# %%
def objective(trial: optuna.Trial) -> float:
    """
    Optuna 目标函数 - 结合总回报与最大回撤的优化目标。
    高回报和低回撤将获得更高的评分。
    移除SMA和ADX的权重参数。
    """
    trial_start_time = time.time()
    
    # --- 建议参数范围 ---
    params = {
        'rsi_length': trial.suggest_int('rsi_length', 5, 20),
        'iv_length': trial.suggest_int('iv_length', 60, 250, step=10), 
        'macd_fast': trial.suggest_int('macd_fast', 5, 15),
        'macd_slow': trial.suggest_int('macd_slow', 15, 35),
        'adx_length': trial.suggest_int('adx_length', 10, 25),  # 保留ADX计算，但不用于权重评分
        'volume_ma_period': trial.suggest_int('volume_ma_period', 10, 50),
        
        # Volume Spike参数
        'volume_spike_window': trial.suggest_int('volume_spike_window', 10, 30),
        'volume_spike_mult': trial.suggest_float('volume_spike_mult', 1.5, 3.0, step=0.25),

        'rsi_low_percentile': trial.suggest_int('rsi_low_percentile', 20, 40),
        'rsi_high_percentile': trial.suggest_int('rsi_high_percentile', 60, 80),
        'iv_low_percentile': trial.suggest_int('iv_low_percentile', 20, 40),
        'iv_high_percentile': trial.suggest_int('iv_high_percentile', 60, 80),

        'take_profit_mult': trial.suggest_float('take_profit_mult', 0.5, 5.0, step=0.25),
        'stop_loss_mult': trial.suggest_float('stop_loss_mult', 0.5, 5.0, step=0.25),

        # 双阈值差额机制
        'bull_score_threshold': trial.suggest_float('bull_score_threshold', -0.1, 0.7, step=0.05), # 允许负值作为多头阈值
        'bear_score_threshold': trial.suggest_float('bear_score_threshold', -0.7, -0.1, step=0.05), # 负值范围，从大到小

        # --- 移除SMA和ADX权重，保留其他权重参数 ---
        'rsi_bull_weight': trial.suggest_float('rsi_bull_weight', 0.2, 0.5, step=0.05),
        'iv_bull_weight': trial.suggest_float('iv_bull_weight', 0.1, 0.4, step=0.05),
        'macd_bull_weight': trial.suggest_float('macd_bull_weight', 0.2, 0.5, step=0.05),
        'volume_bull_weight': trial.suggest_float('volume_bull_weight', 0.05, 0.3, step=0.05),

        'rsi_bear_weight': trial.suggest_float('rsi_bear_weight', 0.2, 0.5, step=0.05),
        'iv_bear_weight': trial.suggest_float('iv_bear_weight', 0.1, 0.4, step=0.05),
        'macd_bear_weight': trial.suggest_float('macd_bear_weight', 0.2, 0.5, step=0.05),
        'volume_bear_weight': trial.suggest_float('volume_bear_weight', 0.05, 0.3, step=0.05),
    }
    # 确保 MACD 慢线 > 快线
    if params['macd_slow'] <= params['macd_fast']:
        params['macd_slow'] = params['macd_fast'] + trial.suggest_int('macd_slow_diff', 1, 10) # 保证慢线大于快线

    # 固定 MACD 信号周期（减少参数量）或也进行优化
    params['macd_signal'] = 9 # 固定为 9 或 trial.suggest_int('macd_signal', 5, 15)

    try:
        # 运行回测 - 设置verbose=False以减少输出信息
        portfolio, _ = run_backtest(price_data, params, initial_capital=INITIAL_CAPITAL, pyramiding=PYRAMIDING, pct_equity=PCT_OF_EQUITY, verbose=False)
        stats = portfolio.stats()

        # 获取目标指标值
        total_return = stats['Total Return [%]']
        max_drawdown = stats['Max Drawdown [%]']
        
        # 判断是否有效交易
        total_trades = stats['Total Trades']
        
        # 结合总回报和最大回撤的综合评分
        # 使用主要指标(总回报)并惩罚大的回撤
        # 总回报权重为0.8，最大回撤权重为0.2
        # 回撤为负数，所以用减法
        if total_trades > 0:
            # 检查NaN值并处理
            if np.isnan(total_return) or np.isnan(max_drawdown):
                # 减少输出，避免过多的日志
                return -1000.0
                
            # 综合评分 = 总回报 - (最大回撤权重/回报权重) * 最大回撤
            # 这里用0.25作为回撤权重比例，相当于回撤的20%权重
            combined_score = total_return - 0.25 * max_drawdown
            
            # 再次检查确保不返回NaN值
            if np.isnan(combined_score):
                # 减少输出，避免过多的日志
                return -1000.0
            
            # 不再在这里打印，由callback处理显示
            return combined_score
        else:
            # 如果没有交易，给出一个较差的分数，但不输出信息
            return -1000.0

    except Exception as e:
        # 仅在关键错误时才输出
        return -1000.0

# %% [markdown]
# ## 10. 执行 Optuna 优化 (Run Optuna Optimization)
# 创建 `optuna` 研究对象，并运行优化过程。

# %%
# --- Optuna Study ---
print(f"--- 开始 Optuna 参数优化 ({N_TRIALS} trials) ---")
print(f"优化目标: 综合指标(总回报[80%] + 最大回撤[20%])")

# 由于我们的目标是最大化综合评分，direction设为'maximize'
study = optuna.create_study(direction='maximize')

# 获取CPU核心数并显示
print(f"系统CPU核心数: {cpu_count}")
cpu_to_use = max(1, cpu_count - 1)  # 保留一个核心给系统
print(f"将使用 {cpu_to_use} 个CPU核心进行并行优化")

# 创建进度条和时间估计
print("\n开始优化，请耐心等待...")
start_time = time.time()
trials_completed = 0

# 定义回调函数来更新进度 - 改进版本，使用sys.stdout实现更清晰的更新
def callback(study, trial):
    global trials_completed, start_time
    trials_completed += 1
    elapsed_time = time.time() - start_time
    avg_time_per_trial = elapsed_time / trials_completed
    estimated_remaining = avg_time_per_trial * (N_TRIALS - trials_completed)
    
    # 计算进度百分比
    progress = trials_completed / N_TRIALS * 100
    
    # 创建进度条
    bar_length = 30
    filled_length = int(bar_length * trials_completed / N_TRIALS)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # 格式化时间
    remaining_str = str(timedelta(seconds=int(estimated_remaining)))
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
    
    # 使用sys.stdout实现更好的进度更新
    # 先检查是否发现了新的最优解(通过best_value变化检测)
    best_found = False
    if trial.value is not None and study.best_trial.number == trial.number:
        best_found = True
        
    # 构建进度显示
    progress_line = f"\r进度: [{bar}] {progress:.1f}% 完成 | 试验: {trials_completed}/{N_TRIALS} | 已用: {elapsed_str} | 剩余: {remaining_str}"
    
    # 如果找到新的最优解，在新行显示
    if best_found:
        sys.stdout.write("\n")
        sys.stdout.write(f"发现新最优: Trial {trial.number} - Score: {trial.value:.2f}\n")
    
    # 更新进度条
    sys.stdout.write(progress_line)
    sys.stdout.flush()

# 使用joblib的parallel_backend来优化并行设置
with parallel_backend('loky', n_jobs=cpu_to_use):
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

# 打印新行以防止覆盖
print("\n")
print(f"优化完成! 总用时: {str(timedelta(seconds=int(time.time() - start_time)))}")

print("\n--- Optuna 优化完成 ---")
print(f"最佳 Trial 编号: {study.best_trial.number}")
print(f"最佳 综合评分: {study.best_value:.2f}")
print("最佳参数:")
best_params = study.best_params

# 创建完整参数字典，添加固定参数，但不添加ADX权重和SMA参数
best_params_complete = best_params.copy()
# 添加固定的MACD信号值
best_params_complete['macd_signal'] = 9
# 不再添加SMA参数，因为已经完全移除了SMA计算

# 按照更易读的格式打印参数
for key, value in best_params.items():
    print(f"  {key}: {value}")

# 添加固定参数信息
print(f"  macd_signal: {best_params_complete['macd_signal']}")


# %% [markdown]
# ## 11. 使用最优参数进行最终回测 (Final Backtest with Best Parameters)
# 使用 `optuna` 找到的最佳参数，重新运行一次回测，并展示详细结果。

# %%
print("--- 开始最终回测 (使用优化后的最佳参数) ---")
final_backtest_start = time.time()
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(final_backtest_start))}")

# 执行最终回测 - 设置verbose=True以显示详细信息
final_portfolio, final_indicator_df = run_backtest(price_data, best_params_complete, verbose=True)
final_backtest_end = time.time()
final_backtest_duration = final_backtest_end - final_backtest_start
print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(final_backtest_end))}")
print(f"回测用时: {str(timedelta(seconds=int(final_backtest_duration)))}")

print("--- 最终回测结果 (优化后) ---")
final_stats = final_portfolio.stats()
print(final_stats)

# 打印关键指标
print("\n关键指标 (优化后):")
print(f"  开始时间: {final_stats['Start']}")
print(f"  结束时间: {final_stats['End']}")
print(f"  持续时间: {final_stats['Period']}")  # 使用'Period'代替'Duration'
print(f"  初始资本: {final_stats['Start Value']:.2f}")
print(f"  最终资本: {final_stats['End Value']:.2f}")
print(f"  总回报率 [%]: {final_stats['Total Return [%]']:.2f}%")
print(f"  最大回撤 [%]: {final_stats['Max Drawdown [%]']:.2f}%")
print(f"  最大回撤持续时间: {final_stats['Max Drawdown Duration']}")
print(f"  综合评分: {final_stats['Total Return [%]'] - 0.25 * final_stats['Max Drawdown [%]']:.2f}")

# 使用try-except处理可能不存在的键
try:
    if 'Annualized Return [%]' in final_stats:
        print(f"  年化回报率 [%]: {final_stats['Annualized Return [%]']:.2f}%")
    # vectorbt 0.27.2版本可能使用不同的键名
    elif 'Ann. Return [%]' in final_stats:
        print(f"  年化回报率 [%]: {final_stats['Ann. Return [%]']:.2f}%")
    else:
        # 手动计算年化回报率（简化版本）
        total_return = final_stats['Total Return [%]']
        days = (final_stats['End'] - final_stats['Start']).days
        years = days / 365.0
        annualized_return = ((1 + total_return/100)**(1/years) - 1) * 100 if years > 0 else total_return
        print(f"  年化回报率 [%] (手动计算): {annualized_return:.2f}%")
except Exception as e:
    print(f"  无法获取或计算年化回报率: {e}")

print(f"  夏普比率: {final_stats['Sharpe Ratio']:.2f}")
print(f"  索提诺比率: {final_stats['Sortino Ratio']:.2f}")
print(f"  胜率 [%]: {final_stats['Win Rate [%]']:.2f}%")
print(f"  总交易次数: {final_stats['Total Trades']}")


# %% [markdown]
# ## 12. 可视化最终回测结果 (Visualize Final Backtest Results)
# 使用 `plotly` 绘制优化后策略的回测表现图。
# %%
print("\n--- 生成最终回测图表 ---")
try:
    # 创建基本图表 - 不指定子图类型，让 vectorbt 默认决定
    fig_final = final_portfolio.plot()

    # 创建一个新的独立图表来显示评分
    fig_final_scores = go.Figure()
    
    fig_final_scores.add_trace(
        go.Scatter(
            x=final_indicator_df.index, 
            y=final_indicator_df['bullish_score'],
            name='多头评分', 
            line=dict(color='green')
        )
    )
    
    fig_final_scores.add_trace(
        go.Scatter(
            x=final_indicator_df.index, 
            y=final_indicator_df['bearish_score'],
            name='空头评分', 
            line=dict(color='red')
        )
    )
    
    # 添加评分差额线
    fig_final_scores.add_trace(
        go.Scatter(
            x=final_indicator_df.index, 
            y=final_indicator_df['score_difference'],
            name='评分差额', 
            line=dict(color='blue')
        )
    )

    # 添加阈值线
    fig_final_scores.add_hline(
        y=best_params_complete['bull_score_threshold'], 
        line=dict(color='green', dash='dash'), 
        name='多头差额阈值'
    )

    fig_final_scores.add_hline(
        y=best_params_complete['bear_score_threshold'], 
        line=dict(color='red', dash='dash'), 
        name='空头差额阈值'
    )

    # 添加零轴线
    fig_final_scores.add_hline(
        y=0, 
        line=dict(color='gray', dash='dot'), 
        name='零轴'
    )

    # 更新主图布局
    fig_final.update_layout(
        title=f"{TICKER} 优化后回测结果 ({START_DATE} to {END_DATE}) - Best Value: {study.best_value:.2f} ({OPTIMIZATION_METRIC})"
    )
    
    # 更新评分图布局
    fig_final_scores.update_layout(
        title=f"{TICKER} 优化后评分指标 ({START_DATE} to {END_DATE})",
        xaxis_title="日期",
        yaxis_title="评分",
        height=400
    )
    
    # 显示两个图表
    fig_final.show()
    fig_final_scores.show()
except Exception as e:
    print(f"最终回测可视化错误: {e}")
    print("继续执行其他过程...")


# %% [markdown]
# ## 13. Optuna 优化历史可视化 (Visualize Optuna Optimization History)
# (可选) 使用 `optuna.visualization` 来查看优化过程。

# %%
if optuna.visualization.is_available():
    print("\n--- 生成 Optuna 优化历史图表 ---")
    # fig_opt_history = optuna.visualization.plot_optimization_history(study)
    # fig_opt_history.show()

    try:
        # 参数重要性图
        fig_param_importances = optuna.visualization.plot_param_importances(study)
        fig_param_importances.show()
    except Exception as e:
        print(f"无法生成参数重要性图: {e}")

    try:
        # 参数关系图 (可能需要 matplotlib)
        # fig_slice = optuna.visualization.plot_slice(study)
        # fig_slice.show()
        pass # plot_slice 可能比较慢或复杂
    except Exception as e:
         print(f"无法生成参数切片图: {e}")

else:
    print("Optuna 可视化不可用。请安装 matplotlib: pip install matplotlib")

# 同时在最终的结果中展示完整的最佳参数
print("\n最终优化参数结果:")
# 创建一个完整的参数字典，但排除ADX权重
final_params = {k: v for k, v in best_params_complete.items() if not (k.startswith('adx_') and k.endswith('_weight'))}
# 按照更易读的格式打印参数
for key, value in final_params.items():
    print(f"  {key}: {value}")

print("="*50)
print(f"脚本总执行时间: {str(timedelta(seconds=int(time.time() - total_script_start_time)))}")
print_total_execution_time()
print("="*50) 