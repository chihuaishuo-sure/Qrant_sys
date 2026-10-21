# 🛡️ Qrant —— 先求少亏的个人量化风控系统

[![CI](https://github.com/OWNER/Qrant_sys/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/Qrant_sys/actions/workflows/ci.yml)

基于 Python + Streamlit 的个人量化工作台。核心理念：**不预测涨跌，而是用有学术证据的规则控制亏损**，并用一整套统计检验识别“回测好看、实盘亏钱”的假优势。

一套工具覆盖个人量化研究的完整闭环：

```
取数（多源回退+复权） → 验数（质量检查） → 回测（A股规则撮合） → 证伪（样本外 / PBO / 随机择时 / PSR）
      → 出信号（每日自动） → 监控（周度回归对比、季度样本外）
```

## 一键启动（Windows）
1. 安装 Python 3.10+（https://www.python.org/downloads/ ，安装时勾选 **Add python.exe to PATH**）
2. 双击 **`启动.bat`**
   - 首次运行会自动创建虚拟环境，并通过清华镜像安装依赖，约 2–5 分钟
   - 之后每次双击，几秒内浏览器就会打开 http://localhost:8501
3. 关闭黑色命令行窗口即退出

macOS / Linux：`./run.sh`

## 功能
| 页面 | 用途 |
|---|---|
| 🏠 今日信号 | 收盘后查看自选 ETF 的目标仓位、市场状态、建议止损价，次日开盘执行；数据有问题会直接警告 |
| 📈 策略回测 | 四线对比（买入持有 / 裸策略 / 风控策略 / 同仓位基准），验收区同时给出**随机择时检验**和 **PSR** |
| 🧺 组合配置 | 多资产反波动率配置 + 趋势过滤 + 组合波动目标，给出每只 ETF 买多少钱、多少股 |
| 🔬 稳健性检验 | 参数热力图、Walk-Forward 样本外、Deflated Sharpe、**回测过拟合概率 PBO**、蒙特卡洛回撤分布 |
| 🛡️ 风控工具箱 | 仓位计算器、VaR/CVaR、交易前检查清单、亏损恢复表 |

## 自动化
| 命令 | 做什么 |
|---|---|
| 双击 `安装定时任务.bat` | 注册 Windows 定时任务：每个工作日 16:30 自动运行 `auto_run.py auto` |
| `python auto_run.py daily` | 刷新行情 → 数据质量检查 → 今日信号 + 组合建议持仓 → 列出“需要操作”的变化，写入 `reports/daily/` |
| `python auto_run.py weekly` | A股验证，并自动对比上一份报告（区分“数据变化”和“代码退化”）；`auto` 模式下每周五运行 |
| `python auto_run.py quarterly` | 样本外检验，监控策略是否衰减；`auto` 模式下每季度首周周五运行 |
| `python compare_reports.py --strict` | 对比最新两份验证报告，有代码退化时返回码 1 |

**推送到手机（可选）**：在项目根目录新建 `notify.json`（不会上传到 GitHub），例如 `{"type": "serverchan", "key": "SCT..."}`，支持 Server酱、企业微信、钉钉、飞书，格式见 `qrant/notify.py`。

**GitHub Actions**：每次推送自动跑单元测试 + 演示数据流水线。海外服务器通常连不上 A 股行情接口，所以真实数据验证在本机由定时任务完成。

系统**只出信号、不下单**；也**不会自动调参改默认值**（那是过拟合机器）。

## 回测真实性
- t 日收盘出信号，**t+1 日开盘成交**（无前视偏差，有单元测试保证）
- 佣金（最低 5 元）、过户费、卖出印花税（ETF 免）、滑点、100 股整手、涨跌停无法成交
- 行情前复权：东方财富前复权优先；限流时回退新浪，并按新浪复权因子处理分红与份额折算
- 数据质量检查：涨跌幅超过涨跌停、价格异常、长时间缺口、连续零成交、数据过期

## 怎么判断一个策略“真的有用”
| 检验 | 回答的问题 | 出处 |
|---|---|---|
| 同仓位基准 | 优势是择时带来的，还是只因为仓位低？ | PRD v1.1 |
| 随机择时检验 | 同样的仓位节奏，随机平移后还能这么好吗？ | 参考 qsx-strategy-score |
| Walk-Forward | 只用过去数据挑参数，未来表现如何？ | Pardo |
| PBO（CSCV） | “挑最好的参数”这件事，本身比随机挑更好吗？ | Bailey 等 2015 |
| Deflated Sharpe / PSR | 扣掉运气、偏度、肥尾后，真实夏普 > 0 的概率？ | Bailey & López de Prado 2012/2014 |
| 试验登记簿 | 一共试了多少次？（`reports/试验登记.csv`） | Harvey & Liu 2015 |

## 推荐用法
1. 以「组合配置」为主力：默认 6 只 ETF（沪深300、中证500、创业板、黄金、国债、纳指）
2. 每月最后一个交易日收盘后查看建议持仓，次日调仓（或安装定时任务，每天自动生成报告）
3. 个股交易前先用「仓位计算器」，单笔最多亏账户的 1%
4. 想改参数时，先跑 `python validate_oos.py`：样本外不胜出就不要改

## 项目结构
```
app.py               界面
auto_run.py          自动化入口（daily / weekly / quarterly / auto）
validate_ashare.py   A股真实数据批量验证 → reports/
validate_oos.py      参数方案样本外检验（标的外 / 策略外 / 时间切分 / Walk-Forward / 配对检验 / PBO）
compare_reports.py   两份验证报告的回归对比
qrant/
  data.py            AKShare 东财→新浪多源回退 + 复权 + 本地缓存 + CSV 导入 + 演示数据
  quality.py         行情数据质量检查
  strategies.py      多周期动量 / 均线趋势 / 唐奇安突破 / RSI 回调 / 趋势组合信号
  risk.py            波动率目标 / 大盘过滤 / ATR 止损 / 仓位计算 / VaR
  backtest.py        A股规则撮合引擎（含回撤熔断）
  portfolio.py       多资产配置
  validation.py      Walk-Forward / DSR / PSR / PBO / 配对检验 / 随机择时 / 块自助蒙特卡洛
  compare.py         报告回归对比
  registry.py        试验登记簿
  notify.py          可选消息推送
docs/                调研对比、PRD v1.0、评审与 v1.1、验证报告、v1.2 研究改进与自动化
tests/               单元测试（python -m pytest tests -q）
reports/             验证报告、样本外检验、试验登记
```

## 常见问题
- **数据获取失败**：东方财富接口偶尔限流，系统会自动尝试新浪接口和本地缓存；也可在侧边栏切换到“演示数据”。
- **双击启动.bat 后弹出微软商店**：说明没有正确安装 Python，请从 python.org 安装并勾选 PATH。
- **想换自选标的**：在「今日信号」页修改代码列表即可。
- **取消定时任务**：双击 `卸载定时任务.bat`。

> ⚠️ 本系统是风险管理与研究工具，不构成投资建议。历史回测不代表未来收益；趋势策略在震荡市会连续小亏，这是避开大跌的代价。
