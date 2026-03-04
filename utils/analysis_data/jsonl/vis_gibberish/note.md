# 训练日志“乱码/退化”诊断指标说明（中文）

本文档说明我们在 veRL/DAPO 训练日志（rollout JSONL）上计算的一组**传统统计指标**，用于在**不调用 LLM**的前提下，刻画并定位训练中常见的“**一本正经地胡说** / **乱码** / **机械复读** / **数值发散**”等异常。每个指标都给出**精确定义**、**计算公式/步骤**、**取值范围与直觉解释**，并标注与训练崩溃的典型关联。

---

## 0. 记号与聚合

* 一条样本（记录）记为 (r)，其**应答序列长度（token 数）**为 (T_r)（仅统计 response 段）。
* **按 step 聚合**：第 (s) 步的样本集合为 (\mathcal{R}_s)。某个样本级指标 (m(r)) 的 step 级聚合：

  * **均值**：(\displaystyle \text{mean}_s=\frac{1}{|\mathcal{R}*s|}\sum*{r\in\mathcal{R}_s} m(r))
  * **中位数**：(\text{median}_s=\text{median}{m(r):r\in\mathcal{R}_s})
  * **求和**：(\displaystyle \text{sum}*s=\sum*{r\in\mathcal{R}_s} m(r))

> 画图时：横轴 (s=\text{global_step})，纵轴为上述聚合。底层散点显示该 step 的**所有样本值**（“沙箱”）。
正确错误分类：优先用 --correct-key 指定的路径（如 reward_extra_infos.acc），否则回退到 reward_seq_sum>0

---

## 1. 概率/对数似然家族（CE、KL、熵）

### 1.1 交叉熵（CE）——Self 与 Ref

* **样本级 token 平均 CE（Self）**
  对当前策略 (\pi_\theta)，样本 (r) 的 response token 记为 (y_{1:T_r})，条件为上下文 (x)：
  [
  \text{CE}^{\text{self}}*{\text{mean}}(r)
  = -\frac{1}{T_r}\sum*{t=1}^{T_r}\log \pi_\theta!\left(y_t\mid x,y_{<t}\right)\quad(\text{nats})
  ]
* **样本级 token 求和 CE（Self）**
  [
  \text{CE}^{\text{self}}*{\text{sum}}(r)
  = -\sum*{t=1}^{T_r}\log \pi_\theta!\left(y_t\mid x,y_{<t}\right)
  ]
* **Ref 版本**：把 (\pi_\theta) 换成参考策略 (\pi_{\text{ref}})。

**含义与直觉**

* CE 越大，表示“**模型对自己输出越不自信/越意外**”。
* 若 **Self-CE 飙升而 Ref-CE不明显** → 模型**偏离参考**、自我退化。
* **mean 与 sum**：mean 近似“单位 token 难度”，sum 还受长度影响（长序列更容易把 sum 拉大）。

### 1.2 KL 近似（mean log-ratio）

* 我们用**蒙特卡洛近似**（沿样本序列）：
  [
  \widehat{D_{\mathrm{KL}}}\big(\pi_\theta\parallel \pi_{\text{ref}}\big)(r)
  ;\approx;
  \frac{1}{T_r}\sum_{t=1}^{T_r}\Big[\log \pi_\theta(y_t\mid\cdot)-\log \pi_{\text{ref}}(y_t\mid\cdot)\Big]
  ]
* 实际记录的就是上式的**token 平均 log-ratio**。

**含义与直觉**

* 估计了“**当前策略相对参考的偏移**”。
* **Self-CE↑ 且 KL↑**：强烈提示“**策略漂移**”而非数据偶发。
* 注意：这是**沿采样轨迹**的近似，不等价于严格 KL，但趋势敏感。

### 1.3 Token 熵（mean）

* 训练时若记录了每个 response token 的预测分布熵 (H_t)：
  [
  \text{Entropy}*{\text{mean}}(r)=\frac{1}{T_r}\sum*{t=1}^{T_r}H_t,
  \quad H_t=-\sum_{w}p_t(w)\log p_t(w)
  ]
  **直觉**
* **机械复读**→ 熵**显著下降**；
* **乱码/噪声**有时会让熵**异常升高或波动**；需与其它重复/字符集指标联判。

---

## 2. 长度与规模

### 2.1 response_len_tokens

* 训练侧直接给出。表示 response 段 token 数 (T_r)。

### 2.2 resp_char_len / resp_byte_len

* **字符长度**：(\text{resp_char_len}(r) = |s|)（Python 字符串长度）。
* **字节长度**：以 UTF-8 编码并采用 `errors="replace"`，
  [
  \text{resp_byte_len}(r)=\big|\text{UTF8_encode}(s,\text{replace})\big|.
  ]
  **直觉**
* 突然变长/极长 often 触发**数值不稳**（sum 类指标被拉大、采样温度副作用等）。

---

## 3. 重复与退化（Degeneracy）

### 3.1 unique_word_ratio（唯一词比例）

* 以**空白分词**（`s.split()`）得到词序列 (w_{1:n})（过滤空串）。
  [
  \text{unique_word_ratio}(r)=\frac{|{w_i}|}{n}.
  ]
  **直觉**：**机械复读**→ 比例**下降**；语料/语言不同对基线值有影响，趋势更重要。

### 3.2 top1_word_prop（最高频词占比）

[
\text{top1_word_prop}(r)=\frac{\max_{w}#{i:w_i=w}}{n}.
]
**直觉**：卡在某个词（如 therefore/所以）→ 该占比**明显升高**。

### 3.3 repeat_char_run_max / repeat_word_run_max（最长重复游程）

* **字符级**：
  [
  \text{repeat_char_run_max}(r)=\max_k\big{\text{存在连续 }k\text{ 个相同字符}\big}.
  ]
* **词级**同理。
  **直觉**：出现 `aaaaaa`/`))))))`/同一词连刷 → **值变大**，对“发疯串”最敏感。

### 3.4 trigram_repeat_ratio（三元词重复率）

* 仍以空白分词得 (w_{1:n})。当 (n\ge3)，构造三元词序列：
  [
  \mathcal{T}=\big{(w_i,w_{i+1},w_{i+2})\big}*{i=1}^{n-2},\quad N*{\text{tg}}=|\mathcal{T}|=n-2.
  ]
* 统计每个三元词的频次 (c(t))。**我们采用“重复出现总量”的比率**：
  [
  \text{trigram_repeat_ratio}(r)=
  \frac{\displaystyle \sum_{t\in\mathcal{T}} \mathbf{1}[c(t)>1]\cdot c(t)}{N_{\text{tg}}}.
  ]

  > 注意：若某三元词出现 3 次，贡献的是 **3** 而不是 (3-1)。因此该比率衡量“**重复三元词在全部三元词中的占比（计出现次数）**”。
* 若 (n<3)，定义为 0。

**直觉**

* 比**游程**更“模式敏感”：即使不连续复读，只要**同样的三词片段**在段落内反复出现，该比率也会抬升。
* 对“**节律性复读**”“模板句换壳复用”尤其敏感。

### 3.5 compress_ratio（压缩比）

* 把字符串以 UTF-8 `errors='replace'` 编码为字节流 (B)，用 **zlib/DEFLATE（LZ77+Huffman）** 压缩：
  [
  \text{compress_ratio}(r)=\frac{|\text{zlib_compress}(B,\text{level}=6)|}{|B|}.
  ]

  * (|\cdot|) 为字节数；空串时定义为 1.0。
  * **DEFLATE** 先用滑动窗口匹配重复段（LZ77），再做哈夫曼编码；因此**可重复、可预测**的文本→**更易压缩**→**比值更小**。

**直觉与注意**

* **机械复读**、模板化产出 → 压缩比显著**降低**（越小越可压缩）。
* **极短字符串**可能因为头部开销出现 (\text{ratio}\ge1)，不代表异常。
* 与“字符熵”相关但不等价：压缩 ratio 还反映了**可被算法捕捉到的结构性重复**。

---

## 4. 字符集与编码异常

### 4.1 non_ascii_ratio（非 ASCII 比例）

[
\text{non_ascii_ratio}(r)=\frac{#{,\text{字符 }c:\ \mathrm{ord}(c)>127,}}{|s|}.
]
**直觉**

* **中文/数学/多语**天然较高；**单看此值不能断言乱码**，需与 4.2/4.3 联判。
* 英文任务若突然升高，可能切换语系或“脏字符”混入。

### 4.2 replacement_char_ratio（替换字符比例）

* **替换字符为** `U+FFFD`（黑菱形问号 `�`），表示**“不能正确解码/映射的未知码点被替换”**。
  [
  \text{replacement_char_ratio}(r)=\frac{#{c=\text{U+FFFD}}}{|s|}.
  ]
  **为什么会出现 `�`？**
* 上游 **字节 → 字符** 解码时，遇到**非法字节序列**而启用了 `"replace"` 策略；
* 中间环节做了编码-解码往返（例如日志系统、外部接口），非法片段被统一替成 `�`；
* 生成端产生了**孤立代理项**或**未定义码点**，在某次再编码时被替换。

**直觉**：这是**编码异常的强信号**。`non_ascii_ratio` 高但 `replacement_char_ratio` 低，多半是“正常非 ASCII 文本”；`replacement_char_ratio` 升高极值得关注。

### 4.3 control_private_unassigned_ratio（控制/私用/未分配类比例）

* 依据 **Unicode General Category**，对每个字符取 `unicodedata.category(c)`：

  * “控制/其他”类以 **`C*`** 打头；其中常见：

    * `Cc` 控制字符（如 `\x00`, `\x1b`）
    * `Cf` 格式控制
    * `Cs` 代理项
    * `Co` 私用区（PUA）
    * `Cn` 未分配
* 统计这些类别的占比：
  [
  \text{control_private_unassigned_ratio}(r)=\frac{#{c:\ \text{category}(c)\in\text{C* 或 Co/Cs/Cn}}}{|s|}.
  ]
  **直觉**：在自然语言输出中这类字符通常**极少见**。异常升高常见于**编码错误**、**二进制数据混入**或**模型退化乱码**。

---

## 5. 信息论指标

### 5.1 char_entropy_bits_per_char（字符级香农熵）

* 基于字符频率 (p(c))：
  [
  H_{\text{char}}(r)=-\sum_{c}p(c)\log_2 p(c),\quad p(c)=\frac{#{c}}{|s|}.
  ]
  **直觉**
* **极度重复** → 熵**很低**；
* **随机噪声/乱码** → 熵可能**偏高**（但配合 4.2/4.3/3.5 更准）。
* 与 token 熵（1.3）不同：这里是**字符频率**，不依赖模型分布。

---

## 6. 样本分类（着色）与可视化

* **样本判定**：优先读取 `--correct-key` 指定路径（如 `reward_extra_infos.acc`），否则回退到
  [
  \mathbf{1}[\text{reward_seq_sum}(r)>0]
  ]
  来近似 **correct / wrong**；未知为 **no_judge**。
* **绘图**：所有样本以散点显示（按类别着色），每个 step 上叠加黑色折线（均值/中位数/求和）。

---

## 7. 经典模式的判读指南

| 异常模式        | 指标联动特征（样本级→step 聚合会显著）                                                                                                                              |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **机械复读/卡词** | `unique_word_ratio↓`、`top1_word_prop↑`、`repeat_*_run_max↑`、`trigram_repeat_ratio↑`、`compress_ratio↓`、（常伴随）`entropy_seq_mean↓`、`CE_self↑`、`reward↓`  |
| **乱码/异常字符** | `replacement_char_ratio↑`（**关键**）、`control_private_unassigned_ratio↑`、（常伴随）`non_ascii_ratio↑`、`char_entropy_bits_per_char` 不稳定、`CE_self↑`、`reward↓` |
| **过长诱发不稳**  | `response_len_tokens↑`（常伴随 `resp_char_len↑`、`CE_self_sum↑`、`reward↓`），有时 `entropy_seq_mean` 和 `KL` 也抬                                               |
| **偏离参考策略**  | `CE_self↑` 而 `CE_ref` 变化不大；`KL(mean log-ratio)↑` 明显                                                                                                 |

> **注意**：`non_ascii_ratio` 高**不是**乱码的充分条件（中文/公式会自然升高）。出现 `�`（U+FFFD）或大量 `C*` 类字符，才是编码层面“红灯”。

---

## 8. 实践建议

1. 优先画 **FAST** 数值（CE/KL/熵/长度）找“**什么时候开始坏**”。
2. 在坏窗口内再看 **MEDIUM** 与 **HEAVY**（重复/压缩/字符集），判断“**坏的性质**”。
3. 若 `replacement_char_ratio` 或 `control_private_unassigned_ratio` 抬头，优先排查**编码链路**（日志落盘、接口中转、UTF-8/UTF-16/代理项）。
4. 若是复读退化，在 **token-mean** 下更常见：可从采样温度/长度上限/惩罚项（重复惩罚、n-gram block 等）和 **loss_agg_mode** 的选择上做对照实验；同时关注 **dropped 比例**与 **prompt 内 reward 方差**是否变化。

---

## 9. 指标一览（速查表）

| 指标                                 | 定义/公式（简）                                               | 取值/趋势          | 典型含义        |   |   |     |      |
| ---------------------------------- | ------------------------------------------------------ | -------------- | ----------- | - | - | --- | ---- |
| `CE self mean/sum`                 | (-\frac{1}{T}\sum\log\pi_\theta) / 求和                  | ↑变差            | 模型对自身输出不自信  |   |   |     |      |
| `CE ref mean/sum`                  | 把 (\pi_\theta) 换 (\pi_{\text{ref}})                    | —              | 参考策略基线      |   |   |     |      |
| `KL mean log-ratio`                | (\frac{1}{T}\sum(\log\pi_\theta-\log\pi_{\text{ref}})) | ↑偏离            | 偏离参考策略      |   |   |     |      |
| `entropy_seq_mean`                 | token 熵平均                                              | 复读↓ / 乱码波动     | 生成不确定性      |   |   |     |      |
| `response_len_tokens`              | response token 数                                       | 过长↑            | 易引起不稳       |   |   |     |      |
| `resp_char_len/byte_len`           | 字符/字节长度                                                | —              | 辅助长度视角      |   |   |     |      |
| `unique_word_ratio`                | 唯一词/总词                                                 | 复读↓            | 多样性         |   |   |     |      |
| `top1_word_prop`                   | 最高频词/总词                                                | 复读↑            | 卡词程度        |   |   |     |      |
| `repeat_char/word_run_max`         | 最长同字符/词游程                                              | 复读↑            | 连续重复        |   |   |     |      |
| `trigram_repeat_ratio`             | 重复三元词出现数 / 全部三元词数                                      | 复用↑            | 模板/节律复用     |   |   |     |      |
| `compress_ratio`                   | (                                                      | \text{zlib}(B) | /           | B | ) | 复读↓ | 可压缩性 |
| `non_ascii_ratio`                  | 非 ASCII 字符占比                                           | 语言/乱码↑         | 多语/数学/或异常   |   |   |     |      |
| `replacement_char_ratio`           | `U+FFFD` 占比                                            | 乱码↑（强信号）       | 解码/码点异常     |   |   |     |      |
| `control_private_unassigned_ratio` | C*/Co/Cs/Cn 占比                                         | 乱码↑（强信号）       | 控制/私用/未分配字符 |   |   |     |      |
| `char_entropy_bits_per_char`       | (-\sum p\log_2 p)                                      | 复读↓/噪声↑        | 字符层信息量      |   |   |     |      |

---

以上指标在我们的脚本中**严格按上述定义实现**，并在每个 step 上提供 **均值/中位数/求和** 三种聚合方式。建议先用 **均值**扫全局，再用 **中位数**验证是否“厚尾带歪”，最后用 **求和**观察“长度+数量”共同作用。
