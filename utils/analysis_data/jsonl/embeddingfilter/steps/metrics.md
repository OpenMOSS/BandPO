# RL 训练日志分析：指标说明文档

## 0. 总览

我们对每条 sample（一次 RL rollout 的一个 `response`）提取三大类特征：

1. **基础统计 + 文本质量**（字符级、token 级、CE / KL / reward 等）
2. **分段 / 分句 + discourse marker + 语言分布**
3. **基于 embedding 的语义结构指标**

   * 语义冗余 / 车轱辘话（R_dup, cluster, plateau）
   * 语义轨迹是否绕圈（loopiness）
   * 组合成一个 RamblingScore（车轱辘话打分）

后续的筛选 & 可视化都是基于这些 metrics。

> 下文所有指标默认都是 **per-sample（每条 response）** 的统计。

---

## 1. 基础 sample-level 指标

这些指标主要来自 `build_features_preprocess.py` 中的 `text_metrics`，以及 JSON 里的原始 scalar。

### 1.1 索引相关

* **`sample_idx`**

  * 含义：样本的全局索引（从 0 开始），约等于 `line_no-1`。
  * 用途：用于在多个 parquet（features / segments / embeddings）之间对齐。
  * 示例：第 100 行 JSON 对应 `sample_idx=99`。

* **`line_no`**

  * 含义：样本所在的原始 JSONL 的行号（1-based）。
  * 示例：第 1 行 `line_no=1`。

* **`byte_offset` / `byte_len`**

  * 含义：样本在 JSONL 文件中的字节偏移和长度（用于 random access）。
  * 计算：顺序读文件时记下 `f.tell()` 和 `len(line)`。
  * 示例：某一行从第 123456 字节开始，长度 789 字节，则 `byte_offset=123456, byte_len=789`。

---

### 1.2 训练相关数值

来自 JSON 字段 `old_logprob_*`, `ref_logprob_*`, `reward_seq_sum`, `kl_seq_mean_log_ratio`, `entropy_seq_mean` 等：

* **`ce_self_seq_mean` / `ce_self_seq_sum`**

  * 含义：自模型（policy）的平均 / 总 cross-entropy，按 token mean（取负 logprob）。
  * 计算：

    * `ce_self_seq_mean = - old_logprob_mean`
    * `ce_self_seq_sum = - old_logprob_sum`
  * 示例：

    * 若 `old_logprob_mean = -1.5`，则 `ce_self_seq_mean = 1.5`（越大说明自模型越不确定）。

* **`ce_ref_seq_mean` / `ce_ref_seq_sum`**

  * 含义：参考模型（ref）的 cross-entropy。
  * 计算：同上，把 `ref_logprob_*` 取负。

* **`reward_seq_sum`**

  * 含义：整条 response 的 reward 之和（你在 RL 框架里定义的 reward）。
  * 示例：数学题答对 reward=1，答错 reward=0，那 `reward_seq_sum` 通常就是 1 或 0。

* **`kl_seq_mean_log_ratio`**

  * 含义：平均 KL（log ratio）指标（policy vs ref），来自训练日志。
  * 示例：值大说明近期更新偏离 ref 较多。

* **`entropy_seq_mean`**

  * 含义：policy 的平均 token-level entropy（越大越“发散”）。

* **`response_len_tokens`**

  * 含义：response token 长度（你原本的统计）。

* **`cls_id`**

  * 含义：样本标签：

    * `1` = 正确（正确答案）
    * `0` = 错误
    * `2` = 无法判断
  * 计算：优先用 `--correct-key` 指定的字段；否则在 `reward_extra_infos` 中找 `acc/is_correct` 等；再不行 fallback 到 `reward_seq_sum>0`。

---

### 1.3 文本质量 & 乱码指标（response_text）

对 `response_text` 做字符级统计：

* **`resp_char_len` / `resp_byte_len`**

  * 含义：字符长度 / UTF-8 字节长度。
  * 示例：字符串 `"你好A"` 字符长度=3，字节长度=7（两个汉字 3*3 +1）。

* **`non_ascii_ratio`**

  * 含义：非 ASCII 字符比例 = 非 ASCII 数 / 总字符数。
  * 示例：

    * `"hello"` → 0
    * `"你好"` → 1
    * `"hello你好"` → 2/7 ≈ 0.2857

* **`replacement_char_ratio`**

  * 含义：Unicode replacement char（`\uFFFD`）占比，用于检测解码错误/乱码。
  * 计算：`count('\uFFFD') / resp_char_len`。

* **`control_private_unassigned_ratio`**

  * 含义：Unicode 类别为控制符/私有区/未分配的字符占比（category 以 `C` 开头）。
  * 用途：检测隐藏控制字符、无效 Unicode、私有区杂质。

* **`repeat_char_run_max`**

  * 含义：最长的“同一字符连续出现”的 run 长度。
  * 示例：

    * `"aaaaabb"` → 5
    * `"哈哈哈哈哈？"` → 5
    * `"haaaaaa ha"` → 6

* **`repeat_word_run_max`**

  * 含义：按空格 split 后，最长的“同一个词连续重复次数”。
  * 示例：

    * `"yes yes yes no"` → 3
    * `"42 42 42 42"` → 4

* **`unique_word_ratio`**

  * 含义：不同词的占比 = `unique(words)/len(words)`。
  * 示例：

    * `"a b c"` → 1
    * `"a a a a"` → 1/4 = 0.25

* **`top1_word_prop`**

  * 含义：最频繁词的频率比。
  * 示例：

    * `"a a a b"` → 3/4 = 0.75

* **`trigram_repeat_ratio`**

  * 含义：三元词组的重复比率（粗抓“模板化 / spam”）。
  * 计算：

    * 按词构造所有 trigram，统计出现次数 >1 的 trigram 总出现次数 / trigram 总数。
  * 示例：

    * `"a b c a b c"` → trigram: (a,b,c), (b,c,a), (c,a,b) … 只要 (a,b,c) 多次出现，则 ratio 提高。

* **`compress_ratio`**

  * 含义：`zlib` 压缩率 = `len(compressed_bytes) / len(raw_bytes)`。
  * 直觉：越接近 0 说明重复结构越多（更容易压缩），越接近 1 说明字符分布更“随机”。

* **`char_entropy_bits_per_char`**

  * 含义：基于字符频率的香农熵（比特/字符）。
  * 计算：
    [
    H = - \sum_c p(c) \log_2 p(c)
    ]
  * 示例：

    * `"aaaaaa"` → 熵约 0
    * `"abcdef"`（全不同） → 熵较高（~2.58 bits/char）

* **`contain_chinese`**

  * 含义：`prompt_text` 或 `response_text` 中是否出现中文（CJK 范围）：

    * `1` = 有中文
    * `0` = 没有
    * `-1` = 无法判断（无文本）

---

## 2. 分段 / 分句 & discourse marker 指标

### 2.1 Section & sentence

按 `response_text` 做两级切分：

1. **section**：按 `\n+` 切（多个连续 `\n` 当成一个 split point，并附着在前一段）。
2. **sentence**：在每个 section 内用 `pysbd` 英文句子切分（不做清洗，原样字符串）。

#### 指标

* **`num_sections`**

  * 含义：一个 response 被 `\n+` 切成的段数。
  * 示例：

    * `"line1\n\nline2\nline3"` → sections=3。

* **`num_sentences`**

  * 含义：所有 section 中的句子总数（pysbd 的结果）。

* **`num_marker_sentences`**

  * 含义：带任意 discourse marker（SELF_REFLECT/CONTRAST/CONCLUDE）的句子数量。

* **`num_self_reflect_sentences` / `num_contrast_sentences` / `num_conclude_sentences`**

  * 含义：分别统计三个类别的句子数。
  * SELF_REFLECT 示例关键字：

    * `wait`, `hold on`, `let me think`, `等一下`, `подожди`, `moment mal`, ...
  * CONTRAST：

    * `however`, `but`, `另一方面`, `jedoch`, `однако`, ...
  * CONCLUDE：

    * `so`, `therefore`, `因此`, `综上`, `итак`, ...

* **`num_marker_sections`**

  * 含义：至少包含一个 marker 句子的 section 数量。

> 后续在 analysis 里会用这些计数算一些比例，如
> `r_self_sent = num_self_reflect_sentences / num_sentences`。

---

### 2.2 sentence-level 辅助字段（在 `segments.parquet`）

每个句子在 `segments.parquet` 中一行，主要字段：

* `sample_idx` / `section_idx` / `sent_idx`
* `char_start` / `char_end`：该句在原始 response 中的字符 span。
* `text`：句子原文。
* `sent_char_len`：句子字符数。
* `marker_mask`：

  * bit 0：SELF_REFLECT
  * bit 1：CONTRAST
  * bit 2：CONCLUDE
* `marker_self_reflect` / `marker_contrast` / `marker_conclude`：0/1 标记。
* `marker_wait_like`：是否命中 `wait` / `hold on` / `等一下` / `подожди` 等更狭义的“打断再想想”。

---

## 3. 语言分布相关指标（Stage 3）

在 per-sample 的 sentence 子集中，基于 `lang_main` 和 `sent_char_len` 计算。

### 3.1 `lang_main` / `lang_conf`（sentence-level）

* 在 preprocess 中，每个句子调用 `langid.classify(text)` 得到：

  * `lang_main`: 语言代码（例如 `en/zh/de/ru/it/...`，过短的句子 → `unk`）
  * `lang_conf`: 模型置信度（0~1）

---

### 3.2 `H_lang`

* 含义：response 的**语言熵**（按字符数加权）。
* 计算步骤：

  1. 对句子按 `lang_main` 聚合字符数：
     [
     w(\ell) = \frac{\text{该语言所有句子字符数之和}}{\text{全 response 字符总数}}
     ]
  2. 熵：
     [
     H_{\text{lang}} = - \sum_\ell w(\ell) \log w(\ell)
     ]
* 直觉：

  * 单一语言（几乎全英文）：(H_{\text{lang}}) 接近 0。
  * 多语言混杂：(H_{\text{lang}}) 较大。

### 3.3 `num_lang_major`

* 含义：占比 > 0.1 的语言种类数量。
* 示例：

  * `w(en)=0.8, w(zh)=0.15, w(ru)=0.05` → `num_lang_major=2`（en, zh）。

### 3.4 `code_switch_count`

* 含义：按句子顺序统计 `lang_main` 改变的次数。
* 示例：

  * 句子语言序列：`[en, en, zh, zh, en]`
    → switches: en→zh (1), zh→en (1) → `code_switch_count=2`

---

## 4. embedding 冗余 / cluster / 语义轨迹指标（Stage 3）

在 Stage 2 中，每个句子被编码为归一化 embedding (\mathbf{e}_t \in \mathbb{R}^d)：

[
|\mathbf{e}_t|_2 = 1
]

因此内积 (\mathbf{e}_p^\top \mathbf{e}_q \in [-1, 1]) 就是 cos 相似度。

对某个 sample 的 N 个句子，我们构造 Gram 矩阵：

[
G_{pq} = \mathbf{e}_p^\top \mathbf{e}_q
]

并在不同子集上进行统计：

* **all**：全句子子集
* **marker**：`marker_mask != 0` 的句子子集
* **wait**：`marker_wait_like == 1` 的句子子集

每类子集都计算一组指标：

* `*_r_dup`
* `*_cluster_count`
* `*_cluster_size_max`
* `*_rho_max`
* `*_loopiness`
* `*_plateau_len_max`

下文以某个子集为例介绍这些概念。

---

### 4.1 冗余（近重复比例）：`*_r_dup`

* 对子集有 m 个句子，embedding 向量 (\mathbf{z}_1, ..., \mathbf{z}*m)，对应的子矩阵 (G*{ij} = \mathbf{z}_i^\top \mathbf{z}_j)。

* 对每个 (i)，定义最大邻接相似度：
  [
  s^{\text{NN}}*i = \max*{j\ne i} G_{ij}
  ]

* 对一个阈值 (\tau_{\text{dup}})（默认 0.9）：

  [
  R_{\text{dup}} = \frac{1}{m} \sum_{i=1}^m \mathbf{1}\left(s^{\text{NN}}*i \ge \tau*{\text{dup}}\right)
  ]

* 直觉：

  * 大量语义上几乎一样的句子，`R_dup` 会接近 1。
  * 正常 CoT 中，句子各自贡献新的信息，`R_dup` 较低。

---

### 4.2 cluster（语义类簇）：`*_cluster_count`, `*_cluster_size_max`, `*_rho_max`

* 在子集上构造一个无向图：

  * 节点：句子
  * 若 `G[i,j] >= cluster_threshold`（默认 0.85），连一条边。

* 取该图的连通分量作为 cluster：

  * `cluster_count`：簇的数量。
  * `cluster_size_max`：最大簇大小。
  * `rho_max = cluster_size_max / m`：最大簇占比。

* 直觉：

  * 正常 CoT：多个不同话题 / 推理阶段 → `cluster_count` 较多、`rho_max` 不太大。
  * 车轱辘话：某一块语义反复绕圈，形成 **一个超大簇** → `cluster_size_max` & `rho_max` 很大。

---

### 4.3 语义轨迹绕圈：`*_loopiness`

对子集中的句子，按 `(section_idx, sent_idx)` 排序成序列：

[
\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_N
]

* 邻接弧长（单位球面上的弧角）：
  [
  d_t = \arccos(\mathbf{e}*t^\top \mathbf{e}*{t+1})
  ]

* 语义路径长度：
  [
  L = \sum_{t=1}^{N-1} d_t
  ]

* 起点和终点的位移：
  [
  D = \arccos(\mathbf{e}_1^\top \mathbf{e}_N)
  ]

* **loopiness** 定义为：
  [
  \lambda = \frac{L}{D + \epsilon}, \quad \epsilon = 1e{-4}
  ]

* 直觉：

  * 如果语义按某个方向稳定推进（越说越不一样），(L \approx D)，(\lambda \approx 1)。
  * 如果在某个主题周围绕来绕去，走了很多“弯路”，最后又回到类似的地方：(L) 很大但 (D) 不大 → (\lambda \gg 1)。

---

### 4.4 语义停滞段：`*_plateau_len_max`

仍然在同一个子集上，对 cluster 序列考察：

1. 已有 cluster_labels（连通分量标号），按 `(section_idx, sent_idx)` 排序得到序列：
   [
   c_1, c_2, ..., c_N
   ]

2. 统计其中最长的“连续相同簇”的段长：

   * `plateau_len_max` = 最大的连续 `c_t` 相同的 run 长度。

* 直觉：

  * 正常 CoT：同一 cluster 内会有几句连着说，但不会特别长。
  * 车轱辘话：往往出现很长的 plateau（比如同一 cluster 下连续 6~10 句）。

---

### 4.5 各子集的具体字段

**all 句子子集：**

* `all_r_dup`
* `all_cluster_count`
* `all_cluster_size_max`
* `all_rho_max`
* `all_loopiness`
* `all_plateau_len_max`

**所有 marker 句子子集（`marker_mask != 0`）：**

* `marker_r_dup`
* `marker_cluster_count`
* `marker_cluster_size_max`
* `marker_rho_max`
* `marker_loopiness`
* `marker_plateau_len_max`

**“wait-like” 句子子集（`marker_wait_like == 1`）：**

* `wait_r_dup`
* `wait_cluster_count`
* `wait_cluster_size_max`
* `wait_rho_max`
* `wait_loopiness`
* `wait_plateau_len_max`

---

### 4.6 `embedding_truncated`

* 含义：是否对该样本做了句子子采样（避免 N 太大导致 O(N²) 内存/时间爆炸）。
* 取值：

  * `0.0`：未截断
  * `1.0`：句子数超出 `--max-sent-per-sample`，只取了等间隔的子集。

---

## 5. RamblingScore（车轱辘话打分）

最后，我们把一些关键指标组合成一个标量：

* 自我打断密度：`r_self_sent = num_self_reflect_sentences / num_sentences`
* 冗余程度：`marker_r_dup`
* marker 区域的 loopiness：`marker_loopiness`
* marker 区域 plateau 长度：`marker_plateau_len_max`
* 语言混乱度：`H_lang`

通过加权线性组合得到：

[
\text{RamblingScore} =
w_{\text{self}} \cdot r_{\text{self}}

* w_{\text{dup}} \cdot \text{marker_r_dup}
* w_{\text{loop}} \cdot \underbrace{\frac{\min(\text{marker_loopiness}, L_{\max})}{L_{\max}}}_{\text{裁剪归一化}}
* w_{\text{plateau}} \cdot \mathbf{1}(\text{marker_plateau_len_max} \ge k)
* w_{\text{lang}} \cdot H_{\text{lang}}
  ]

其中：

* `w_*` 可以通过 args 调整（默认都 1，语言权重 0.1）。
* `L_max` = `rambling_loopiness_cap`（比如 5）。
* `k` = `rambling_plateau_k`（比如 4）。

**直觉：**

* **高** `RamblingScore` 对应：

  * 自我打断句子很多（`wait/hold on/等一下` 类占比高）
  * 这些 marker 句子在语义上高度冗余（`marker_r_dup` 高）
  * 语义轨迹在 marker 区域绕圈（`marker_loopiness` 大）
  * 有长 plateau（反复说同一类东西）
  * 且语言熵较高（多语言混杂）

* **低** `RamblingScore`：正常 CoT，偶尔自我纠错，语义推进比较直接，没怎么兜圈子。

---

## 6. 一个简单示例（定性说明）

假设有两条 response：

### 示例 A：正常简短推理

```text
"2+2=4. Therefore the answer is 4."
```

预期：

* `num_sections ≈ 1`，`num_sentences ≈ 2`。
* marker：

  * `num_conclude_sentences ≈ 1`（有 "Therefore"）。
  * `num_self_reflect_sentences = 0`。
* 语言：

  * `H_lang ≈ 0`（几乎全英文），`code_switch_count = 0`。
* embedding：

  * all 子集：句子不重复，`all_r_dup` ~ 0，`all_loopiness` ~ 1。
  * marker 子集只有一条句子：`marker_r_dup=0`，`marker_loopiness=1`，`marker_plateau_len_max=1`。
* RamblingScore：各项都低 → 得分很低 → 非车轱辘。

---

### 示例 B：典型车轱辘话 + 自我打断

```text
"Wait, let me think. 
Actually, wait, I think I made a mistake.
Hold on, I'm not sure.
But on the other hand, maybe it's 4.
However, I still feel uncertain.
So, so, so, the answer is probably 4.
The answer is 4. The answer is 4."
```

预期：

* 分句后有 6~7 个句子。
* marker：

  * `num_self_reflect_sentences` 高（`Wait`, `Actually, wait`, `Hold on`）。
  * `num_contrast_sentences` 也有 (`But`, `However`)。
  * `num_marker_sentences` / `num_sentences` 比例较高。
* 语言：

  * 全英文但自我打断频繁，`H_lang` 稍微 >0 但不大，`code_switch_count` ~ 0（这里主要是单一语言）。
* embedding：

  * 多个句子都在“我不确定”“再想想”这个语义附近，形成一个大 cluster。

    * `marker_r_dup` 较高，`marker_rho_max` 接近 1。
    * `marker_plateau_len_max` 可能 >=4（连续很多句子属于同一个 cluster）。
  * 语义轨迹：

    * 前半部分围绕“不确定”绕圈，后半部分掉到“答案是 4”附近。
    * `marker_loopiness` 会显著 >1。
* RamblingScore：

  * `r_self_sent` 高、`marker_r_dup` 高、`marker_loopiness` 大、`plateau>=k`、`H_lang` 中等 → 得分显著高 → 车轱辘嫌疑非常大。

---

如果你后面在可视化阶段画一些 scatter（比如 `ce_self_seq_mean` vs `rambling_score`、`reward_seq_sum` vs `marker_r_dup`），就可以直观看到：

* 哪些“答对的样本”（reward 高）同时 `RamblingScore` 非常高 → 优先过滤的候选；
* 哪些“训练崩溃前”的 batch 中，`loopiness` / `R_dup` 远高于整体均值。

这就是整个 metrics 设计的核心用途。
