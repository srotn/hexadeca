# Hexadeca

**Hexadeca 是一个原创棋类与计算机博弈 AI 研究项目。**

项目研究一个确定性的 16×16 空间策略游戏，并提供从规则引擎、神经网络、蒙特卡洛树搜索到自对弈训练和对战评估的完整实验平台。项目重点是让一个结构清晰、可复现实验的原创棋类成为计算机博弈研究对象。

## 游戏概览

- 16×16 非环绕棋盘，黑方先行，双方交替落子。
- 每回合在一个空格放置一枚棋子；棋子不移动、不吃子、不替换。
- 候选格必须与所有已有棋子保持 Chebyshev 距离大于 1，因此任意棋子都会禁止其自身及周围八邻格。
- 没有随机事件、隐藏信息、Pass 或同时行动。
- 没有合法落点时立即终局。
- 终局对全部 256 个格点进行距离层计分，首个黑白数量不相等的欧氏距离层决定该格归属。
- 256 个可能的动作位置；由 2×2 分块可知理论最大长度为 64 plies。

正式规则见 [`docs/rules/hexadeca-v1.md`](docs/rules/hexadeca-v1.md)。规则版本 `hexadeca-v1` 不随本次开源包装改变。

## AI 系统

当前 16×16 实现包括：

- Policy Network：输出 256 个动作 logits，并在搜索时使用合法动作掩码；
- Value Network：预测当前行棋方的结果以及双方归一化终局分数；
- 残差 CNN 主干与 D4 数据增强；
- MCTS / PUCT；
- First Play Urgency（FPU）；
- Virtual Loss；
- Batch Leaf Selection 与 batched neural inference；
- Dirichlet 根噪声和温度采样，用于自对弈探索；
- Replay 数据、AdamW 训练和 checkpoint 管理；
- 可选 C++20 Native 棋盘、计分、特征和搜索内核；
- 自对弈、换色对战、置信区间和相对 Elo 评估。

精确残局模块目前用于小规模剩余局面的 WDL 求解，不代表 16×16 已被完全求解。当前公开版本也不附带训练 checkpoint、Replay buffer 或实验运行目录。

## Visual results

这些图展示了当前公开实现的 16×16 研究结果。它们来自历史 checkpoint 的固定配置演示，不代表完整的统计评估。

### Five checkpoints, five terminal positions

每个 checkpoint 各进行一局无噪声、每步 1600 simulations 的同 checkpoint 自对弈。棋盘颜色表示终局计分归属，圆点表示实际落子；下方给出五度特征和棋局长度、分差。

<p align="center">
  <img src="docs/assets/hexadeca-five-checkpoint-comparison.png" alt="Five Hexadeca checkpoints and their terminal positions" width="100%">
</p>

### Example terminal analysis

下面是一局 `iteration-004520` 的终局分析：37 plies，蓝方 129 分，橙方 127 分。图中同时展示终局棋盘、局部格度热图、逐点散度/侵度热图和五度参数。

<p align="center">
  <img src="docs/assets/hexadeca-analysis-iteration-004520.png" alt="Terminal analysis of a Hexadeca game from iteration 004520" width="100%">
</p>

### Internal relative Elo

下图使用已有的 16×16 checkpoint round-robin 结果绘制。每个点是 Bradley–Terry 相对等级分，阴影是 95% bootstrap 区间；`iteration-001360` 固定为 1000。评估使用每步 800 simulations、颜色平衡的 checkpoint 对战。该 Elo 只用于项目内部比较，不能与国际象棋、围棋或其他游戏的等级分直接比较。

<p align="center">
  <img src="docs/assets/hexadeca-relative-elo.png" alt="Hexadeca internal relative Elo across training checkpoints" width="100%">
</p>

## 复杂度摘要

下表是基于当前 `hexadeca-v1` 规则的组合计数与结构分析，不是程序 benchmark，也不是对所有状态逐一枚举后的运行时间测量。

| 量 | 结果 |
|---|---:|
| Maximum game length | 64 plies |
| Action space | 256 |
| Reachable colored states | ≈ 1.56896 × 10^46 |
| Uncolored legal occupied-set states | ≈ 4.42220 × 10^34 |
| Full game-tree history nodes | ≈ 2.04243 × 10^104 |
| State-space information complexity | ≈ 153.46 bits |
| Game-tree information complexity | ≈ 346.51 bits |
| State-space peak | 44 plies |
| Peak states at 44 plies | ≈ 2.09534 × 10^45 |
| Game-tree peak | 63 plies |
| Opening branching factor | 256 |
| Effective average branching factor | ≈ 42.63 |

The state-space peak and game-tree peak occur at different depths: around 44 plies for distinct states and around 63 plies for historical game-tree nodes.

Selected late-game layers are approximately:

| Layer | States |
|---|---:|
| ≤1 ply from theoretical maximum | ≈ 2.896 × 10^34 |
| ≤4 plies | ≈ 2.175 × 10^38 |
| ≤8 plies | ≈ 6.304 × 10^41 |
| ≤10 plies | ≈ 1.127 × 10^43 |
| ≤12 plies | ≈ 1.112 × 10^44 |
| ≤16 plies | ≈ 2.373 × 10^45 |
| 64-piece terminal-layer states | ≈ 4.673 × 10^32 |

推导口径、符号和限制见 [`docs/complexity.md`](docs/complexity.md)。

## 安装

项目需要 Python 3.11–3.13。推荐在虚拟环境中安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Native C++20 扩展是可选加速路径。需要已安装的 C++20 编译工具时，可以执行：

```bash
python setup.py build_ext --inplace --force
```

没有编译扩展时，Python 规则引擎和研究代码仍可用于规则测试、网络测试和 CPU 实验。

## 基本验证

```bash
python -m pytest tests/test_board.py tests/test_environment.py tests/test_scoring.py
python -m pytest
```

完整架构和模块关系见 [`docs/architecture.md`](docs/architecture.md)。测试、格式化和类型检查命令以项目配置为准；CUDA、Native 编译器和 PyTorch 版本会影响可运行的实验范围。

## 目录结构

```text
benchmark/   可复现的规则、网络、搜索、自对弈和训练 benchmark
config/      16×16 规则和训练配置
cpp/         可选 C++20 加速实现
docs/        规则、复杂度和架构说明
game/        棋盘、终局和计分
mcts/        PUCT、节点、策略和残局搜索
network/     输入特征、残差网络、策略与损失
native/      Native 扩展的 Python 接口
tests/       规则、搜索、网络、训练和评估测试
training/    自对弈、Replay、训练和对战评估
utils/       通用工具
```

## 研究状态与范围

这是一个研究型开源项目。已有实现支持训练和自对弈研究，但不宣称给出游戏理论最优策略、绝对人类棋力或跨规则泛化结论。评估中的 Elo 是项目内部相对尺度。20×20 特征实验、训练产物、历史日志和本地运行目录不属于本次 16×16 公共发布范围。

## 许可

本项目使用 MIT License。详见 [`LICENSE`](LICENSE)。
