# Hexadeca 特征解析器

## 目录结构

- `features.py`、`analyze_record.py`、`plot_game_summary.py`：解析与绘图代码
- `records/`：输入棋谱 JSON
- `json/`：特征分析 JSON 输出
- `png/`：终局棋盘、热力图和特征汇总 PNG 输出

解析器读取网页导出的紧凑棋谱 JSON：

```json
{
  "moves": [225, 187, 30],
  "final_matrix": [[...], ...]
}
```

其中 `moves` 使用 `action = row * 16 + column`，`final_matrix` 使用 `1/-1`
表示蓝/橙棋子，`2/-2` 表示蓝/橙领地空位，`0` 表示未归属空位。

## Python 接口

```python
from analysis import HexadecaFeatureParser

parser = HexadecaFeatureParser()
record = parser.parse("hexadeca-game.json")
features = parser.analyze(record)
payload = features.to_dict()
```

也可以直接调用：

```python
from analysis import analyze_game

features = analyze_game("hexadeca-game.json")
```

`features.to_dict()` 返回稳定的 JSON 结构：`features` 中包含 `D`、`G0`--`G2`、
`I`、`S0`--`S2`、`B0`--`B2`；`heatmaps.G0/G1/G2` 中保留每种棋子集合的
`11x11` 疏晶格局部格度矩阵和 `13x13` 密晶格局部格度矩阵；
`point_heatmaps.divergence/intrusion` 分别保存逐点散度和逐点侵度的 `16x16`
浮点矩阵。当前 `schema_version=3`；版本 3 表示侵度采用终局领地符号场定义。

逐点散度把棋盘上的每个坐标依次作为中心，采用 7×7 外接框内、半径为 3 的离散
圆形邻域，即只保留满足 `dr² + dc² <= 9` 的 29 个格点。圆内使用截断高斯权重
`K(dr,dc) = exp(-(dr²+dc²)/(2σ²))`，当前 `σ=2.0`；圆外权重为 `0`。
先计算高斯密度场 `R=K*occupied` 和有效核质量场 `M=K*board_mask`，再对两者分别
施加可分离周期消隐滤波器 `h=[1,3,4,3,1]/12`：
`R_c=h_y*h_x*R`、`M_c=h_y*h_x*M`。该滤波器保留直流项，并将周期 2 与周期 3
的理想振荡响应压为 0，从而降低两种自晶格对散度的影响。逐点值为
`2·[M_c(x,y)/M0]/R_c(x,y)`，其中 `M0` 是完整圆形高斯核质量；中心棋子以权重
`1` 计入密度。为避免滤波尾部在求倒数时产生尖峰，当 `R_c <= 0.1` 时按空邻域
处理，取 `2·M_c/M0`。在所有棋子位置上取平均会严格得到标量 `D`。
绘制 PNG 时不改动上述原始数值，而是采用两个合法终局晶格作为颜色参照端点：
间隔 2 的 64 子致密终局给出下端 `D=0.4607450795`，间隔 3 的 36 子稀疏终局
给出上端 `D=0.8988766847`。热力图值低于或高于该区间时分别映射到色条两端，
并用延伸标记提示越界，从而提高终局棋局间的颜色对比度。
逐点侵度直接使用终局矩阵的符号场，不再使用棋子最近邻距离。定义蓝、橙领地掩码

\[
B(x,y)=\mathbf 1[F(x,y)>0],\qquad
O(x,y)=\mathbf 1[F(x,y)<0].
\]

其中 `1/2` 都属于蓝方，`-1/-2` 都属于橙方，`0` 为中立。为避免单个异号格被
扩散成大块区域，侵度使用比散度更局部的 5×5、半径 2、`σ=1.15` 圆形截断
高斯核 `K`：

\[
b=K*B,\qquad o=K*O,\qquad
q=\frac{K*(B+O)}{K*\mathbf 1_{board}}.
\]

`q` 是有效棋盘范围内的领地覆盖置信度，同时消除边缘核被裁切造成的偏差。领地
交界响应为

\[
C(x,y)=q(x,y)\frac{4b(x,y)o(x,y)}{[b(x,y)+o(x,y)]^2},
\]

当分母为 0 时取 `C=0`。纯蓝区与纯橙区趋近 0，两色各半时响应最高；中立区域
会被 `q` 抑制。

对终局矩阵中每颗真实棋子 `p_i`，按其周围平滑领地计算敌方占比：

\[
e_i=\begin{cases}
\dfrac{o(p_i)}{b(p_i)+o(p_i)},&F(p_i)=1,\\[1ex]
\dfrac{b(p_i)}{b(p_i)+o(p_i)},&F(p_i)=-1.
\end{cases}
\]

再使用半径 2、`σ_p=0.75` 的截断高斯，把打入响应集中在棋子及其紧邻区域：

\[
P(x,y)=1-\prod_i\left[1-e_i
\exp\left(-\frac{\lVert(x,y)-p_i\rVert^2}{2\sigma_p^2}\right)\right].
\]

乘积只包含距离不超过 2 的棋子。最终逐点侵度和标量侵度为

\[
H_I(x,y)=1-[1-C(x,y)][1-P(x,y)],
\qquad
I=\frac{\sum_{x,y}q(x,y)H_I(x,y)}{\sum_{x,y}q(x,y)}.
\]

因此 `H_I` 与 `I` 均严格位于 `[0,1]`，蓝橙领地交界会形成连续响应带，处于大量
敌方领地包围中的棋子会产生额外的平滑亮斑。两张逐点矩阵均覆盖全部 256 个点，
包括 `2/-2` 表示的禁入点。若整个矩阵没有任何已归属格，`I` 输出为 `null`。

局部格度矩阵采用窗口内全局格度算法：疏晶格使用 6×6 窗口，密晶格使用 4×4
窗口，窗口每次移动 1 格。每个窗口都把左上角作为局部坐标原点，重新计算对应的
9/4 个晶格相位；窗口内无棋子时结果为 `0`。因此疏矩阵尺寸为 `11x11`、密矩阵
尺寸为 `13x13`，元素是归一化格度值而不再是同或匹配计数。

侵度现在完全由终局领地符号场和真实棋子位置决定。单方领地的结果为 `0`；只要
矩阵存在已归属领地，即使缺少一方棋子，标量仍有定义。

命令行调用：

```bash
python -m analysis path/to/hexadeca-game.json > features.json
```

推荐使用整理输出命令：

```bash
python -m analysis.analyze_record analysis/records/hexadeca-game.json
```

该命令会自动将结果写入 `analysis/json/` 和 `analysis/png/`。

默认要求着法序列已经终局；分析前端快照时可使用
`--allow-nonterminal`，或在 Python 中传入 `require_terminal=False`。

晶格全局计算默认使用 9 个疏晶格相位 `(0..2, 0..2)` 和 4 个密晶格相位
`(0..1, 0..1)`。如果项目最终确定了其他起点，只需注入：

```python
from analysis import LatticeConfiguration, HexadecaFeatureParser

parser = HexadecaFeatureParser(
    lattice=LatticeConfiguration(
        sparse_origins=((0, 0), ...),
        dense_origins=((0, 0), ...),
    )
)
```
