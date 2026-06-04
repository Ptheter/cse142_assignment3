"""Byte-level BPE tokenizer (no regex pre-tokenization)."""

from __future__ import annotations

from collections import Counter


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """Train a byte-level BPE tokenizer.

    Merge order: at each step, merge the most frequent adjacent pair.
    Break ties by selecting the pair with the smallest ``(id1, id2)``
    in lexicographic (tuple) order.

    IDs 0–255 are single bytes. Merge tokens get IDs starting from 256.
    Special tokens get the highest IDs in the vocab.

    Args:
        input_path: Path to a UTF-8 text file.
        vocab_size: Target vocabulary size (>= 256 + len(special_tokens)).
        special_tokens: Optional special token strings.

    Returns:
        vocab: ``dict[int, bytes]`` mapping token ID to byte string.
        merges: ``list[tuple[int, int]]`` merge pairs in order.
    """
    if special_tokens is None:
        special_tokens = []

    # ---- read data ----
    with open(input_path, "rb") as f:
        text = f.read()

    # ---- initial vocab (byte-level) ----
    vocab = {i: bytes([i]) for i in range(256)}

    # encode text as initial token sequence
    tokens = list(text)

    merges: list[tuple[int, int]] = []

    next_id = 256

    # reserve space for special tokens at the end
    special_start = vocab_size - len(special_tokens)
    special_ids = {
        tok: special_start + i
        for i, tok in enumerate(special_tokens)
    }

    # ---- helper: count pairs ----
    def get_pair_stats(seq):
        counts = Counter()
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
        return counts

    # ---- BPE loop ----
    while next_id < special_start:

        pair_counts = get_pair_stats(tokens)
        if not pair_counts:
            break

        # best pair: max frequency, tie -> smallest lexicographic pair
        best_pair = max(
            pair_counts.items(),
            key=lambda x: (x[1], -x[0][0], -x[0][1])  # frequency desc, id asc
        )[0]

        a, b = best_pair
        merges.append(best_pair)

        # merge tokens
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(next_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens
        vocab[next_id] = vocab[a] + vocab[b]
        next_id += 1

        if len(vocab) >= vocab_size - len(special_tokens):
            break

    # ---- add special tokens at top IDs ----
    for tok, idx in special_ids.items():
        vocab[idx] = tok.encode("utf-8")

    return vocab, merges

class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ) -> None:

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # map merge pair -> rank (earlier merge = higher priority)
        self.merge_ranks = {
            pair: i for i, pair in enumerate(merges)
        }

        # fast lookup: special tokens in text
        self.special_token_set = set(self.special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""

        segments = []
        i = 0

        while i < len(text):
            matched = None

            for tok in self.special_tokens:
                if text.startswith(tok, i):
                    matched = tok
                    break

            if matched is not None:
                segments.append(("special", matched))
                i += len(matched)
            else:
                segments.append(("text", text[i]))
                i += 1

        tokens = []

        for typ, val in segments:

            if typ == "special":
                # SPECIAL TOKEN → single ID (CRITICAL)
                for k, v in self.vocab.items():
                    if v == val.encode("utf-8"):
                        tokens.append(k)
                        break

            else:
                # normal character → bytes
                tokens.extend(list(val.encode("utf-8")))

        while True:
            best_pair = None
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    r = self.merge_ranks[pair]
                    if r < best_rank:
                        best_rank = r
                        best_pair = pair

            if best_pair is None:
                break

            a, b = best_pair

            merged_bytes = self.vocab[a] + self.vocab[b]

            merged_id = None
            for k, v in self.vocab.items():
                if v == merged_bytes:
                    merged_id = k
                    break

            new_tokens = []
            i = 0

            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""

        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")