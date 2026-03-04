
# 特征构建与语义分析流水线说明

本仓库的特征构建分为 **3 个 Python 阶段脚本** + 一个简单的 **bash 封装脚本**：

1. `build_features_preprocess.py`  —— 从 **RL 训练 JSONL** 到  **样本级 + 句子级结构化数据**
2. `build_features_embedding.py`   —— 从 **句子级文本** 到 **embedding 向量**
3. `build_features_analysis.py`    —— 从 **embedding + 各种统计** 到 **最终分析特征**

可选封装脚本：
- `build_features.sh` —— 串联上述三步，形成完整 build pipeline。

下面分别介绍每个脚本的作用、输入输出和主要逻辑。

---

## 0. 整体数据流概览

```text
原始 RL JSONL
  |
  |  (Stage1: build_features_preprocess.py, CPU)
  v
  OUT_DIR/
    ├─ features.parquet          # 每个 sample 一行：原有语言 & reward 特征 + 分段/分句统计 + snapshots
    ├─ segments.parquet          # 每个 sentence 一行：句子文本 + marker + 语言 + 位置
    └─ meta_preprocess.json      # 元信息、schema、参数

  |
  |  (Stage2: build_features_embedding.py, GPU)
  v
  OUT_DIR/
    ├─ sentence_embeddings.parquet   # 每个 sentence 一行：sample_idx + section_idx + sent_idx + emb 向量
    └─ meta_embedding.json

  |
  |  (Stage3: build_features_analysis.py, CPU)
  v
  OUT_DIR/
    ├─ features_analysis.parquet     # 在 features.parquet 基础上增加所有 embedding 分析指标
    └─ meta_analysis.json
````

之后做 **数据筛选、可视化、调 filter 阈值** 时，只需要读 `features_analysis.parquet` 即可。

---

## 1. build_features_preprocess.py

### 1.1 作用概述

**Stage 1：从超大 JSONL 构建基础特征 + 文本切分结构。**

主要工作：

* 一趟顺序扫描 JSONL，记录 `(line_no, byte_offset, byte_len)`，可选计算 sha256。
* 多进程 worker 按 offset 读取每行 JSON，解析出原始字段。
* 保留原先的各类统计特征（CE / entropy / 重复度 / 压缩比 / 乱码指标等）。
* 在 **response_text** 上做两级切分：

  * 按 `\n+` 切 section（保留原始换行）。
  * 对每个 section 用 `pysbd` 做 sentence segmentation。
* 对每个句子做：

  * discourse marker 标注（SELF_REFLECT / CONTRAST / CONCLUDE / wait-like），多语言词表。
  * 句子级语言识别（langid）。
* 汇总出样本级的分段/分句统计（section 数、句子数、marker 句子数等）。
* 输出：

  * `features.parquet`：每 sample 一行。
  * `segments.parquet`：每 sentence 一行。
  * `meta_preprocess.json`：元信息。

### 1.2 输入输出

**输入：**

* 参数 `--jsonl`: RL 训练产生的巨大的 `.jsonl` 文件。
* 参数 `--out-dir`: 输出目录（自动创建）。

**主要输出文件：**

1. `features.parquet`（样本级）

   * 一行对应 JSONL 中一条记录（一个 sample）。
   * 包含：

     * 索引 & 元数据：

       * `sample_idx`：0-based，等于 `line_no - 1`，作为统一主键。
       * `line_no`, `byte_offset`, `byte_len`, `valid`, `uid`, `global_step`, `kept_flag`。
     * 训练相关数值特征（沿用旧代码）：

       * `ce_self_seq_mean`, `ce_self_seq_sum`
       * `ce_ref_seq_mean`, `ce_ref_seq_sum`
       * `reward_seq_sum`, `kl_seq_mean_log_ratio`, `entropy_seq_mean`
       * `response_len_tokens`
       * `cls_id`（1=correct, 0=wrong, 2=no_judge）
     * 文本质量/乱码指标：

       * `resp_char_len`, `resp_byte_len`, `non_ascii_ratio`, `replacement_char_ratio`,
       * `control_private_unassigned_ratio`, `repeat_char_run_max`, `repeat_word_run_max`,
       * `unique_word_ratio`, `top1_word_prop`, `trigram_repeat_ratio`,
       * `compress_ratio`, `char_entropy_bits_per_char`
       * `contain_chinese`（prompt/response 中是否包含中文）
     * snapshot 字段（原始文本等）：

       * 对 `--snapshot-keys` 指定的 dot-path，生成列：

         * 如 `snap__epoch`, `snap__prompt_text`, `snap__response_text`, ...
     * 分段/分句统计：

       * `num_sections`: 按 `\n+` 划分的 section 总数。
       * `num_sentences`: 所有句子的总数。
       * `num_marker_sentences`: 带任意 marker 的句子数。
       * `num_self_reflect_sentences`: SELF_REFLECT 句子数。
       * `num_contrast_sentences`: CONTRAST 句子数。
       * `num_conclude_sentences`: CONCLUDE 句子数。
       * `num_marker_sections`: 至少包含一个 marker 的 section 数。

2. `segments.parquet`（句子级）

   * 一行对应一个 sentence。
   * 核心字段：

     * 对应 sample/位置：

       * `sample_idx`, `line_no`
       * `section_idx`（按 `\n+` 切的 index）
       * `sent_idx`：整条 response 内的句子序号（0-based）。
       * `section_local_sent_idx`：该 section 内的局部句子 index。
     * 位置 & 长度：

       * `char_start`, `char_end`：在原始 `response_text` 里的字符 span。
       * `sent_char_len`：句子长度。
     * 原文：

       * `text`: 句子原文（不清洗）。
     * discourse marker 标记：

       * `marker_mask`：bitmask（bit0=SELF_REFLECT, bit1=CONTRAST, bit2=CONCLUDE）
       * `marker_self_reflect`, `marker_contrast`, `marker_conclude`（0/1）
       * `marker_wait_like`：wait / 等一下 / подожди 等自我打断标记。
     * 语言识别：

       * `lang_main`: langid 检测的主语言（短句/失败则为 `unk`）。
       * `lang_conf`: 语言识别置信度。

3. `meta_preprocess.json`

   * 源文件信息（路径、大小、mtime、sha256）。
   * 输出路径（features/segments）。
   * 行数统计。
   * snapshot key 到列名的映射。
   * 脚本参数（procs, chunk_lines, correct_key, min_lang_chars 等）。

### 1.3 CLI 使用示例

```bash
python build_features_preprocess.py \
  --jsonl /path/to/rollout.jsonl \
  --out-dir /path/to/out_dir \
  --procs 64 \
  --chunk-lines 20000 \
  --correct-key "reward_extra_infos.acc" \
  --snapshot-keys "epoch,global_step,uid,filter_state,prompt_text,response_text,prompt_len_tokens,response_len_tokens,total_len_tokens,reward_seq_sum,reward_extra_infos.acc"
```

---

## 2. build_features_embedding.py

### 2.1 作用概述

**Stage 2：在 GPU 上对所有句子文本做 embedding，得到句子向量。**

主要工作：

* 读取 `segments.parquet`（Stage1 的句子表）。
* 使用一个多语言 embedding 模型（默认 `BAAI/bge-m3`）对每个句子编码：

  * tokenizer + model 编码；
  * mean pooling + L2 normalize → 得到一个 d 维向量。
* 按 row-group 流式处理，避免一次性加载过大。
* 将 `sample_idx/section_idx/sent_idx` 与 `emb` 写入新的 `sentence_embeddings.parquet`，行数与 `segments.parquet` 一一对应。

### 2.2 输入输出

**输入：**

* `--segments-parquet`: `build_features_preprocess.py` 生成的 `segments.parquet`。
* `--out-dir`: 输出目录（通常与 Stage1 的相同）。
* `--model-name`: embedding 模型名（默认 `"BAAI/bge-m3"`）。
* `--device`: 用哪个设备（如 `"cuda"`, `"cuda:0"`, `"cpu"`）。
* `--batch-size`: 每个 batch 的句子数（默认 64）。
* `--max-length`: 模型输入最大 token 长度（默认 512）。
* `--max-segments`: 调试用，限制只编码前 N 条句子（默认 -1 表示全部）。

**输出：**

1. `sentence_embeddings.parquet`

   * 每行对应 `segments.parquet` 的一行（同 row-group & 行号）。
   * 字段：

     * `sample_idx`
     * `section_idx`
     * `sent_idx`
     * `emb`：list[float]，embedding 向量。

2. `meta_embedding.json`

   * 使用的 `segments.parquet` 路径。
   * embedding 模型名、device、batch size、max_length 等参数。
   * 总行数、编码行数。
   * build 时间。

### 2.3 CLI 使用示例

```bash
python build_features_embedding.py \
  --segments-parquet /path/to/out_dir/segments.parquet \
  --out-dir /path/to/out_dir \
  --model-name "BAAI/bge-m3" \
  --device "cuda" \
  --batch-size 64 \
  --max-length 512
```

---

## 3. build_features_analysis.py

### 3.1 作用概述

**Stage 3：基于 embedding 的语义分析与各类统计，生成最终特征。**

主要工作：

* 读取：

  * Stage1 的 `features.parquet`（样本级统计 + text metrics）。
  * Stage1 的 `segments.parquet`（句子级文本 + marker + 语言）。
  * Stage2 的 `sentence_embeddings.parquet`（句子级 embedding）。
* 以 `sample_idx` 为 key，将句子 & embedding 聚合到每个 sample 内，计算：

  * 语言分布统计：`H_lang`、`num_lang_major`、`code_switch_count`。
  * 语义冗余和聚类：

    * `R_dup`：近重复句子比例（基于 cosine 阈值）。
    * cluster 数量、最大 cluster 占比 `rho_max`。
  * 语义轨迹：

    * loopiness（路径长度 / 起终点位移）。
    * plateau（长时间停留在同一 cluster 的段落长度）。
  * 子集版本：

    * 全句子（all_*）
    * marker 句子（marker_*）
    * wait-like 句子（wait_*）
  * 一个可调的 `rambling_score`（车轱辘话程度综合指标）。
* 对每个 sample 写入新的特征列，输出为 `features_analysis.parquet`。

### 3.2 输入输出

**输入：**

* `--preprocess-dir`: Stage1 输出目录（必须包含 `features.parquet` & `segments.parquet`）。
* `--embed-dir`: Stage2 输出目录（必须包含 `sentence_embeddings.parquet`）。
* `--out-dir`: 输出目录（建议与 Stage1/Stage2 相同）。

关键分析参数（全部是可调接口）：

* `--dup-threshold`: 判定近重复的相似度阈值（默认 0.90）。
* `--cluster-threshold`: 聚类连边的相似度阈值（默认 0.85）。
* `--plateau-min-len`: plateau 最小长度（用于统计，默认 3）。
* `--loop-eps`: loopiness 中防除零的小 epsilon（默认 1e-4）。
* `--max-sent-per-sample`: 每个 sample 做 embedding 分析时最大句子数，超过则子采样（默认 128）。
* `--only-kept`: 只对 kept_flag == 1 的样本做 embedding 分析（常用）。
* `--only-correct`: 只对 cls_id == 1 的样本做 embedding 分析（常用）。
* RamblingScore 权重 & 阈值：

  * `--rambling-weight-self`：自我打断比例的权重。
  * `--rambling-weight-dup`：marker 区域重复度的权重。
  * `--rambling-weight-loop`：marker 区域 loopiness 的权重。
  * `--rambling-weight-plateau`：plateau 的权重。
  * `--rambling-weight-lang`：语言熵的权重。
  * `--rambling-loopiness-cap`：loopiness 上限，用于归一化。
  * `--rambling-plateau-k`：认定“长 plateau”的最小长度。

**输出：**

1. `features_analysis.parquet`

   * 基于 Stage1 的 `features.parquet`，逐行增加以下列：

   **语言分布：**

   * `H_lang`：语言熵。
   * `num_lang_major`：占比 > 0.1 的语言种类数。
   * `code_switch_count`：相邻句子语言切换次数。

   **全句子 embedding 指标（all_*）：**

   * `all_r_dup`：近重复句子比例。
   * `all_cluster_count`：cluster 数量。
   * `all_cluster_size_max`：最大 cluster 大小。
   * `all_rho_max`：最大 cluster 占比。
   * `all_loopiness`：loopiness（路径长度 / 起终点位移）。
   * `all_plateau_len_max`：plateau 最大长度。

   **marker 子集 embedding 指标（marker_*）：**

   * 对 marker_mask != 0 的句子做与上面同样的分析：

     * `marker_r_dup`
     * `marker_cluster_count`
     * `marker_cluster_size_max`
     * `marker_rho_max`
     * `marker_loopiness`
     * `marker_plateau_len_max`

   **wait-like 子集 embedding 指标（wait_*）：**

   * 对 `marker_wait_like == 1` 的句子做同样分析：

     * `wait_r_dup`
     * `wait_cluster_count`
     * `wait_cluster_size_max`
     * `wait_rho_max`
     * `wait_loopiness`
     * `wait_plateau_len_max`

   **其它：**

   * `embedding_truncated`：该 sample 是否因为句子过多被子采样（0/1）。
   * `rambling_score`：综合车轱辘话分数（由上面的指标 + Stage1 的 `num_self_reflect_sentences/num_sentences` + `H_lang` 等加权而来）。

2. `meta_analysis.json`

   * 使用的参数（dup-threshold, cluster-threshold 等）。
   * 输入文件路径（features/segments/embeddings）。
   * 处理样本数量、build 时间等。

### 3.3 CLI 使用示例

```bash
python build_features_analysis.py \
  --preprocess-dir /path/to/out_dir \
  --embed-dir /path/to/out_dir \
  --out-dir /path/to/out_dir \
  --dup-threshold 0.90 \
  --cluster-threshold 0.85 \
  --plateau-min-len 3 \
  --max-sent-per-sample 128 \
  --only-kept \
  --only-correct \
  --rambling-weight-self 1.0 \
  --rambling-weight-dup 1.0 \
  --rambling-weight-loop 1.0 \
  --rambling-weight-plateau 1.0 \
  --rambling-weight-lang 0.1 \
  --rambling-loopiness-cap 5.0 \
  --rambling-plateau-k 4
```

---

## 4. build_features.sh（可选封装）

一个简单的 bash 脚本，串联 3 个阶段脚本：

```bash
#!/usr/bin/env bash
set -euo pipefail

JSONL="${1:-/path/to/your.jsonl}"
OUT_DIR="${2:-/path/to/output_dir}"

# Stage1
python build_features_preprocess.py \
  --jsonl "$JSONL" \
  --out-dir "$OUT_DIR" \
  --procs 64 \
  --chunk-lines 20000 \
  --correct-key "reward_extra_infos.acc" \
  --snapshot-keys "epoch,global_step,uid,filter_state,prompt_text,response_text,prompt_len_tokens,response_len_tokens,total_len_tokens,reward_seq_sum,reward_extra_infos.acc"

# Stage2
python build_features_embedding.py \
  --segments-parquet "$OUT_DIR/segments.parquet" \
  --out-dir "$OUT_DIR" \
  --model-name "BAAI/bge-m3" \
  --device "cuda" \
  --batch-size 64 \
  --max-length 512

# Stage3
python build_features_analysis.py \
  --preprocess-dir "$OUT_DIR" \
  --embed-dir "$OUT_DIR" \
  --out-dir "$OUT_DIR" \
  --dup-threshold 0.90 \
  --cluster-threshold 0.85 \
  --plateau-min-len 3 \
  --max-sent-per-sample 128 \
  --only-kept \
  --only-correct \
  --rambling-weight-self 1.0 \
  --rambling-weight-dup 1.0 \
  --rambling-weight-loop 1.0 \
  --rambling-weight-plateau 1.0 \
  --rambling-weight-lang 0.1 \
  --rambling-loopiness-cap 5.0 \
  --rambling-plateau-k 4
```

你可以根据机器的 CPU/GPU 配置和实际需求修改参数（例如把 Stage2 放到 GPU 机器上单独跑、修改 procs/batch size 等）。

---

## 5. 后续工作建议

有了这三个阶段的 build 之后，后续就可以：

* 在 **可视化 / 筛选脚本** 里只读 `features_analysis.parquet`。
* 基于：

  * `rambling_score`
  * `marker_r_dup / marker_loopiness / marker_plateau_len_max`
  * `H_lang / code_switch_count`
  * CE/entropy/乱码指标（沿用原来的）
* 设计各种 filter：比如“reward 高 + RamblingScore 高”的样本先丢弃，缓解 reward hacking 对训练稳定性的影响。

如果你之后想加新的指标（比如更细的 discourse pattern、更多语言支持），建议直接在 Stage1/Stage3 上扩展字段即可，保持同样的数据流设计。
