# Voidsea US Stocks — Public Framework

用于展示美股交易与回测工程结构的 Python 项目。公开版采用 **10/30 SMA 均线交叉示例策略**，ATR 用于示例风控；不包含私人组合信号、研究股票池、账户数据或历史行情。

本项目展示回测、券商适配、订单对账、持久化状态与风险监控。示例参数未经收益优化，不能视为实际收益承诺。

## 快速开始（无需密钥）

Python 3.11+：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/offline_demo.py
python -m unittest discover -s tests -v
```

Windows PowerShell 用 `.venv\Scripts\Activate.ps1` 激活环境。
离线示例在内存中生成合成价格，不下载行情、不访问交易账户、不提交订单。

## 结构

| 目录 | 用途 |
| --- | --- |
| `broker/` | 抽象券商、回测成交、Alpaca 适配 |
| `engine/` | 信号执行、订单对账、账户状态隔离 |
| `market_data/` | 行情缓存、完整 K 线及时效检查 |
| `strategy/` | 通用指标和 SMA 交叉教学示例 |
| `risk/` | ATR 仓位/止损和单日亏损限制示例 |
| `backtest/`、`performance/` | 回测、成本模拟和绩效统计 |
| `tests/` | 合成数据与模拟券商测试 |

示例在短均线上穿长均线时买入，下穿时退出；SELL 表示平仓，不是做空。
示例风控默认单笔风险预算 0.5%、最大仓位 10%、单日亏损限制 2%、固定入场 ATR 的 2 倍止损。它们仅用于演示机制。

## 自行配置行情与 paper 账户

复制 `.env.example` 为项目根目录的 `.env`，填入你自己的 paper key。`.env` 不应提交到 Git。

```bash
cp .env.example .env
python main.py --mode paper
python -m backtest.run_backtest --symbol SPY --days 365 --refresh
```

`main.py` 默认只评估，不下单；它仍然需要访问账户和行情。只有显式传入 `--execute` 才持续执行订单。框架保留 live 适配，live 还需 `TRADING_MODE=live` 和 `ALLOW_LIVE_TRADING=true`；本公开版未对真实交易做验收。

数据缓存、报告、日志和 `data/runtime/` 账户状态必须保留本地。自行确认所用数据的访问和再分发权限。

## 执行限制

- 状态按模式和账户隔离；同一账户的进程需使用同一主机上的同一状态文件。文件锁不协调多台机器。
- 订单接受不等于成交；程序持续对账部分成交、拒单和不确定提交。不要删除状态文件来重试不确定订单，应先人工核对券商记录。
- 止损依赖本地报价轮询，没有券商端保护性止损单。断网、停止进程或轮询间短暂触价可能无法退出；停止程序不会自动平仓。
- 日亏损使用前一交易日权益基准，外部存取款未单独调整。OHLC 回测不能复现实时点差、网络延迟与逐笔成交。
- 缓存文件名的日期表示请求窗口，不保证行情完整；测试使用 mock，不证明券商实盘运行可靠。

## 许可状态

本整理版未替作者新增开源授权。发布者应确认代码来源与贡献权属，再选择许可证并添加 `LICENSE`。在此之前，不将本项目宣称为已获开源许可。依赖许可见 `DEPENDENCIES.md`。
