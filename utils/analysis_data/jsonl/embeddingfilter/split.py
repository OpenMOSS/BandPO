#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentence segmentation using punctuators (SBDModelONNX) while preserving
the original text exactly.

Key idea:
- Use SBD only to get the *sequence of sentences* (rough segmentation).
- Then align each SBD sentence back to the original string via a
  normalization + substring alignment algorithm, and slice the original
  text with [start:end] indices.

This avoids artifacts like the '⁇' character that may appear in SBD outputs
and keeps all original characters untouched.

Dependencies:
    pip install punctuators
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from punctuators.models import SBDModelONNX


# =========================
#  Helpers: char categories
# =========================

def _is_cjk(ch: str) -> bool:
    """Rudimentary CJK check (covering most common ranges)."""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF   # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x20000 <= code <= 0x2A6DF  # CJK Extension B
        or 0x2A700 <= code <= 0x2B73F  # CJK Extension C
        or 0x2B740 <= code <= 0x2B81F  # CJK Extension D
        or 0x2B820 <= code <= 0x2CEAF  # CJK Extension E
        or 0xF900 <= code <= 0xFAFF    # CJK Compatibility Ideographs
        or 0x2F800 <= code <= 0x2FA1F  # CJK Compatibility Ideographs Supplement
    )


def _normalize_for_alignment(text: str) -> Tuple[str, List[int]]:
    """
    Normalize original text into a compact string for alignment, and
    build a mapping from normalized positions to original indices.

    Strategy:
    - Keep only:
        * CJK characters
        * letters (a-zA-Z...) / digits (0-9)
      (including Cyrillic, Greek 等，因为 isalnum() 为 True)
    - Drop whitespace & punctuation (，。！？、:;-"\/$ 等都丢弃)
    - Lowercase latin letters.

    Return:
        norm_str: normalized text
        mapping:  for each position i in norm_str, mapping[i] is the index
                  in the original text where this char came from.
    """
    norm_chars: List[str] = []
    mapping: List[int] = []

    for idx, ch in enumerate(text):
        if ch.isspace():
            continue

        if _is_cjk(ch) or ch.isalnum():
            norm_chars.append(ch.lower())
            mapping.append(idx)
        else:
            # All punctuation / symbols are dropped.
            continue

    return "".join(norm_chars), mapping


def _normalize_sbd_sentence(sent: str) -> str:
    """
    Normalize an SBD output sentence for alignment.

    Almost the same logic as _normalize_for_alignment, but:
    - Additionally drop the '⁇' placeholder if present.
    """
    norm_chars: List[str] = []

    for ch in sent:
        if ch == "⁇":
            # Explicitly drop the double question mark placeholder
            # that sometimes appears in SBD outputs.
            continue
        if ch.isspace():
            continue

        if _is_cjk(ch) or ch.isalnum():
            norm_chars.append(ch.lower())
        else:
            # Drop punctuation / symbols
            continue

    return "".join(norm_chars)


# =========================
#  Core alignment algorithm
# =========================

@dataclass
class SentenceSpan:
    start: int  # inclusive index into original text
    end: int    # exclusive index into original text

    def slice_from(self, text: str) -> str:
        return text[self.start:self.end]


def align_sbd_sentences_to_text(
    text: str,
    sbd_sentences: List[str],
    extend_to_punctuation: bool = True,
) -> List[SentenceSpan]:
    """
    Given original text and SBD sentences (possibly normalized/modified),
    align them back to the original text and return sentence spans
    [start, end) in original indices.

    Args:
        text: Original text (any language mix).
        sbd_sentences: Output of SBDModelONNX.infer([text])[0]
        extend_to_punctuation:
            If True, extend each sentence's end index to include
            trailing sentence-final punctuation (。！？!?;；. 等).

    Returns:
        List[SentenceSpan] in order. Spans should cover the whole text
        without overlap (the last one will be extended to len(text) if
        necessary).
    """
    norm_text, norm2orig = _normalize_for_alignment(text)
    norm_len = len(norm_text)

    spans: List[SentenceSpan] = []
    search_pos = 0

    for sent in sbd_sentences:
        sent_norm = _normalize_sbd_sentence(sent)
        if not sent_norm:
            # Degenerate case, skip empty normalized sentences
            continue

        # Find normalized sentence substring in normalized text,
        # starting from current search_pos to keep order.
        idx = norm_text.find(sent_norm, search_pos)
        if idx == -1:
            # Fallback: try global search (in rare misalign cases)
            idx = norm_text.find(sent_norm)
            if idx == -1:
                # If still not found, skip this sentence.
                # (You could also log a warning here.)
                continue

        start_norm = idx
        end_norm = idx + len(sent_norm)

        # Map normalized span back to original indices
        # "Core" span = from first to last aligned character.
        start_core = norm2orig[start_norm]
        end_core = norm2orig[end_norm - 1] + 1  # inclusive -> exclusive

        # Optionally extend end_core to include trailing punctuation like
        # 。！？!?;；. etc.
        end_ext = end_core
        if extend_to_punctuation:
            # Skip any spaces
            while end_ext < len(text) and text[end_ext].isspace():
                end_ext += 1

            # Include sentence-final punctuation
            sentence_final_punct = "。！？!?；;.!?"
            while end_ext < len(text) and text[end_ext] in sentence_final_punct:
                end_ext += 1

        span = SentenceSpan(start=start_core, end=end_ext)
        spans.append(span)

        # Advance search position in normalized text
        search_pos = end_norm
        if search_pos >= norm_len:
            break

    # Post-process spans to be contiguous & non-overlapping
    final_spans: List[SentenceSpan] = []
    prev_end = 0
    for i, span in enumerate(spans):
        start = max(prev_end, span.start)
        end = max(span.end, start)
        final_spans.append(SentenceSpan(start=start, end=end))
        prev_end = end

    # If there is any remaining tail of the original text not covered
    # by spans, attach it to the last sentence.
    if final_spans:
        if final_spans[-1].end < len(text):
            final_spans[-1] = SentenceSpan(
                start=final_spans[-1].start,
                end=len(text),
            )
    else:
        # No spans at all: fall back to one span covering entire text
        final_spans.append(SentenceSpan(start=0, end=len(text)))

    return final_spans


# =========================
#  Public API
# =========================

class SBDSegmenter:
    """
    A convenience wrapper around SBDModelONNX + alignment.

    Usage:
        seg = SBDSegmenter()
        spans = seg.segment(text)
        sentences = [s.slice_from(text) for s in spans]
    """

    def __init__(self, pretrained_name: str = "sbd_multi_lang") -> None:
        """
        Args:
            pretrained_name: Name supported by SBDModelONNX.from_pretrained,
                             for multilingual SBD use "sbd_multi_lang".
        """
        self.model = SBDModelONNX.from_pretrained(pretrained_name)

    def segment(
        self,
        text: str,
        extend_to_punctuation: bool = True,
    ) -> List[SentenceSpan]:
        """
        Segment a single text into sentence spans (on original text).
        """
        sbd_output: List[List[str]] = self.model.infer([text])
        sbd_sentences = sbd_output[0]
        spans = align_sbd_sentences_to_text(
            text, sbd_sentences, extend_to_punctuation=extend_to_punctuation
        )
        return spans


# =========================
#  Simple test / demo
# =========================

def _build_multilang_math_text() -> str:
    """
    Construct a multi-language math reasoning example in a single string.
    Languages: Chinese + English + German + Russian.
    """
    return (
        "混合数学推理示例：考虑一个 $10 \\times 10$ 的方格棋盘，我们想要放置若干个 L 形三格骨牌，使得它们互不重叠。"
        "First, we observe that the board has 100 cells, and each L-tromino covers exactly 3 cells. "
        "Warte mal, hier gibt es einen Sonderfall: 100 ist nicht durch 3 teilbar, also können wir das ganze Brett nicht vollständig überdecken. "
        "Дальше подождём минутку и подумаем, можно ли хотя бы почти всё поле покрыть, если мы удалим один уголок. "
        "然后我们删去左上角的一个小方格，此时剩下 99 个格子，99 可以被 3 整除，于是从计数上看似乎是有希望的。"
        "However, the coloring argument shows that even after removing one cell, some configurations are still impossible. "
        "所以，我们需要更精细地分析：等一下，我们是不是忽略了棋盘的黑白染色结构？"
    )


def _run_demo() -> None:
    seg = SBDSegmenter(pretrained_name="sbd_multi_lang")

    multi_lang_text = _build_multilang_math_text()
    more_examples = [
        "这是一个纯中文的句子，用来测试多语言 SBD 在中文上的表现。第二句在这里。",
        "This is a pure English example. It contains two sentences.",
    ]

    all_inputs = [multi_lang_text] + more_examples

    for idx, text in enumerate(all_inputs):
        spans = seg.segment(text)

        print("=" * 80)
        print(f"[Input {idx}] 原始文本：")
        print(text)
        print("\n切分结果：")
        for i, span in enumerate(spans):
            sent = span.slice_from(text)
            print(f"  [{i}] ({span.start}, {span.end}) -> {sent}")
        print()


if __name__ == "__main__":
    _run_demo()
