#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多语言 SBD demo（基于 HuggingFace + punctuators）：
- 使用 1-800-BAD-CODE/sentence_boundary_detection_multilang 模型
- 支持在同一个字符串中混合多语言（中文/英文/德文/俄文等）
- 在 CPU 上跑
- 对切完之后仍然“超长”的句子做兜底二次切分
"""

from typing import List
import re

from punctuators.models import SBDModelONNX


def load_sbd_model() -> SBDModelONNX:
    """
    加载多语言 SBD 模型。
    默认就是 CPU + ONNXRuntime，不用管 GPU。
    """
    # 对应 HuggingFace: 1-800-BAD-CODE/sentence_boundary_detection_multilang
    model = SBDModelONNX.from_pretrained("sbd_multi_lang")
    return model


def _fallback_split_long_sentence(
    text: str,
    max_sentence_chars: int = 400,
) -> List[str]:
    """
    对“仍然很长”的句子做兜底切分。

    策略：
    1. 先按强标点切：中英文句号/问号/感叹号/分号。
    2. 每段再检查长度，仍然太长的话按 fixed-size chunk 再切。
    """
    text = text.strip()
    if len(text) <= max_sentence_chars:
        return [text]

    # 1) 按强标点切，但保留标点
    #    re.split 会把括号里的捕获组作为单独元素返回，所以要再拼回去
    parts = re.split(r"([。！？!?；;])", text)
    segments: List[str] = []
    buf = ""
    for chunk in parts:
        if not chunk:
            continue
        buf += chunk
        # 每遇到一个标点，就认为是一个候选句子
        if re.fullmatch(r"[。！？!?；;]", chunk):
            segments.append(buf.strip())
            buf = ""
    if buf.strip():
        segments.append(buf.strip())

    # 2) 对每个 segment 再做长度检查，不合格就按长度粗暴切
    final_segments: List[str] = []
    for seg in segments:
        if len(seg) <= max_sentence_chars:
            final_segments.append(seg)
        else:
            # 粗暴长度切：保证不会出现超级长的一坨
            for i in range(0, len(seg), max_sentence_chars):
                final_segments.append(seg[i : i + max_sentence_chars].strip())

    return final_segments


def sbd_with_fallback(
    model: SBDModelONNX,
    texts: List[str],
    max_sentence_chars: int = 400,
) -> List[List[str]]:
    """
    对一批文本做多语言 SBD，并对超长句子兜底二次切分。

    :param model: SBDModelONNX 实例
    :param texts: List[str] 原始文本（可多语言、可混合）
    :param max_sentence_chars: 单个句子的最大字符数，超过则启动兜底切分
    :return: List[List[str]]，每个输入文本得到一个“句子列表”
    """
    # 使用模型做第一轮切句
    raw_results: List[List[str]] = model.infer(texts)

    # 第二轮：兜底处理超长句子
    final_results: List[List[str]] = []
    for sent_list in raw_results:
        processed: List[str] = []
        for sent in sent_list:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= max_sentence_chars:
                processed.append(sent)
            else:
                processed.extend(
                    _fallback_split_long_sentence(
                        sent, max_sentence_chars=max_sentence_chars
                    )
                )
        final_results.append(processed)

    return final_results


def build_multilang_math_text() -> str:
    """
    构造一个“数学推理 + 多语言混合在同一个 str 里”的测试样例。
    语言：中文 + 英文 + 德语 + 俄语，全部掺在一起。
    """
    text = (
        "混合数学推理示例：考虑一个 $10 \\times 10$ 的方格棋盘，我们想要放置若干个 L 形三格骨牌，使得它们互不重叠。"
        "First, we observe that the board has 100 cells, and each L-tromino covers exactly 3 cells. "
        "Warte mal, hier gibt es einen Sonderfall: 100 ist nicht durch 3 teilbar, also können wir das ganze Brett nicht vollständig überdecken. "
        "Дальше подождём минутку и подумаем, можно ли хотя бы почти всё поле покрыть, если мы удалим один уголок. "
        "然后我们删去左上角的一个小方格，此时剩下 99 个格子，99 可以被 3 整除，于是从计数上看似乎是有希望的。"
        "However, the coloring argument shows that even after removing one cell, some configurations are still impossible. "
        "所以，我们需要更精细地分析：等一下，我们是不是忽略了棋盘的黑白染色结构？"
    )
    return text


def main():
    model = load_sbd_model()

    multi_lang_math_text = build_multilang_math_text()

    # 你也可以顺便测几段单语言文本
    more_examples = [
        "这是一个纯中文的句子，用来测试多语言 SBD 在中文上的表现。第二句在这里。",
        "This is a pure English example. It contains two sentences.",
    ]

    all_inputs = [multi_lang_math_text] + more_examples

    results = sbd_with_fallback(
        model,
        all_inputs,
        max_sentence_chars=400,  # 你可以根据自己任务把阈值调大/调小
    )

    for idx, (input_text, sentence_list) in enumerate(zip(all_inputs, results)):
        print("=" * 80)
        print(f"[Input {idx}] 原始文本：")
        print(input_text)
        print("\n切分结果：")
        for i, s in enumerate(sentence_list):
            print(f"  [{i}] {s}")
        print()


if __name__ == "__main__":
    main()
