# money-machine（刷钱机）

加密货币量化研究和模拟交易系统：用 OKX 的公开行情做研究和**虚拟钱包模拟盘**，强调"先证明有优势，再谈赚钱"。
A crypto quant research + paper-trading system on OKX public data, built around pre-registered
experiments, realistic costs and forward testing.

> **免责声明**：本项目只做研究和模拟交易，不构成任何投资建议。下面所有回测结果都**没有**被证明能在未来赚钱：
> 它们大多没有通过本项目自己设定的统计门槛，前向测试也还在进行中。加密货币交易可能损失全部本金。

## 现在在哪一步

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1.0 | 规则策略 + LLM 策略的对照实验，验证撮合引擎和模拟盘 | 完成；规则策略没有优势，已停用 |
| 2.0 | 横截面机器学习：LightGBM（50 个特征）+ Kronos（K 线基础模型），成本感知的组合优化 | 没过门槛 A；`ML_v2` 自 2026-10-13 起前向模拟 |
| v3 | 12 个预先登记的改进试验 | 都没过门槛 A；信号平滑（A1a）和小资金集中版（B1b）作为探索组前向模拟 |
| v4 | 挂单执行（E）+ 资金费套利（F） | F 没通过；E：集中版改挂单在回放中通过前两条（成本降 42–87%），等服务器上影子挂单实测满 56 天 |
| v5 | BTC/ETH 和前 10 大币种的趋势跟踪 | T1b（BTC+ETH，只做多或空仓）通过全部标准；`T1b_trend` 自 2026-10-16 起前向模拟 |

研究方法：每个试验都先写进研究计划再跑。用带隔离期的滚动验证，锁定 2026-04-01 之后的考试区，
用 Deflated Sharpe 和 PBO 校正多次试验的运气，和同构建方法的随机信号对照，最后看前向模拟盘。

## 研究文档

- [1.0 可行性调研](research/2026-10-08_刷钱机1.0可行性调研.md)
- [2.0 机器学习方案](research/2026-10-08_刷钱机2.0_机器学习方案.md) · [2.0 门槛 A 结果](research/2026-10-12_刷钱机2.0_门槛A结果.md)
- [v3 研究计划](research/2026-10-15_刷钱机v3_研究计划.md) · [v3 结果](research/2026-10-15_刷钱机v3_结果.md)
- [v4 研究计划：挂单执行 + 资金费套利](research/2026-10-15_刷钱机v4_研究计划.md) · [v4 结果](research/2026-10-15_刷钱机v4_结果.md)
- [v5 研究计划：趋势跟踪](research/2026-10-15_刷钱机v5_趋势跟踪研究计划.md) · [v5 结果](research/2026-10-15_刷钱机v5_趋势跟踪结果.md)

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)。数据都来自交易所公开接口，不需要 API key。

```bash
uv sync                                   # 安装依赖
scripts/fetch_kronos.sh                   # 下载 Kronos 源码（MIT，固定版本）到 vendor/
uv run mm ml data                         # 下载 2.0 数据到 store/ml（OKX 公开数据，较慢）
uv run mm ml research                     # 预先登记的滚动验证 + 门槛 A 报告
uv run mm paper --config configs/v2.toml  # 实时模拟盘（Ctrl+C 停止，--resume latest 续跑）
uv run mm report --config configs/v2.toml # 各策略表现 + 放行检查
uv run pytest                             # 测试
```

告警渠道（邮件、PushPlus、飞书）：把 `.env.example` 复制为 `.env` 再填写。

## 部署到服务器（长期运行）

模拟盘要连续跑 8–12 周，建议放在常开的 Linux 服务器上。在**本机终端**运行（需要能 SSH 登录服务器）：

```bash
scripts/deploy_server.sh root@服务器IP          # 首次部署；改了代码或配置后重跑即可更新
scripts/server.sh root@服务器IP status          # 运行状态 + 最新权益
scripts/server.sh root@服务器IP report          # 各策略表现 + 放行检查
scripts/server.sh root@服务器IP logs            # 实时日志
scripts/server.sh root@服务器IP pull            # 账本拉回本地 runs/server_v1.sqlite
uv run mm report --db runs/server_v1.sqlite     # 本地查看拉回的账本
```

部署脚本只会新增：系统用户 `mm`、目录 `/opt/money-machine`（自带 uv 和 Python，不动系统 Python），以及几个 systemd 服务：`money-machine-paper`（模拟盘，内存上限 1.8G）、`money-machine-collect`（盘口和持仓量采集）、`money-machine-execlab`（影子挂单观察，不下单），外加每 5 分钟运行一次的监控 `money-machine-monitor.timer`。所有服务都设了内存和 CPU 上限、低优先级，崩溃后自动续跑。**不开放任何端口**，只主动连接交易所的公开接口。部署前会检查服务器能否访问 OKX、时钟是否准确，以及用户名和服务名有没有冲突。

同一个账本只允许一个模拟盘进程写入（有文件锁），避免两个进程同时写坏状态。

## 它怎么工作

```
OKX 公共 WebSocket（1m K 线）+ 每日数据同步
        │
        ▼
   撮合引擎 + 虚拟钱包（每个策略一个钱包）：手续费 · 滑点 · 资金费 · 强平
        │  每天 00:00 UTC 决策
        ▼
   特征（50 个）→ LightGBM + Kronos → 信号 →（平滑 / 集中持仓）→ 成本感知优化器 → 目标仓位
        │
        ▼
   SQLite 账本 → 报告 + 放行检查（和同构建方法的随机对照组比）→ 监控告警（邮件 / 微信）
```

- 回测和模拟盘用的是同一个引擎；实盘代码和研究代码有逐日一致性测试。
- 撮合规则故意偏保守：
  - 市价单在下一根 K 线开盘成交，再加滑点。
  - 限价单必须被价格穿过才成交。
  - 同一根 K 线里止损和止盈都可能触发时，算止损先触发。

## 目录

```
money-machine/
├── src/moneymachine/     # 引擎、模拟盘、策略、监控
│   └── ml/               #   数据、特征、模型、组合优化、验证、各版本试验（research_v3/v4、carry、trend、execution）
├── configs/              # v1.toml（1.0 对照实验）、v2.toml（当前模拟盘）
├── scripts/              # 部署、服务器运维、Kronos 下载与 GPU 任务
├── research/             # 研究计划和结果报告
├── tests/                # pytest
└── store/ runs/ vendor/  # 数据、账本、第三方代码（不在仓库里，本地生成或下载）
```

## 许可证和第三方

- 本仓库代码以 [MIT](LICENSE) 发布。
- [Kronos](https://github.com/shiyu-coder/Kronos)（MIT）不包含在仓库里，用 `scripts/fetch_kronos.sh` 下载。
- 研究初期参考过的几个第三方项目（GPL/AGPL）和收到的数据归档**没有**放进本仓库，本项目代码也没有引用它们。
- 行情数据来自 OKX 和 Binance 的公开接口，请遵守它们的使用条款。仓库不分发任何数据。
