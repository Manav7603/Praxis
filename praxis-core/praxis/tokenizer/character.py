"""
Character-level tokenizer — one token per Unicode character.

The simplest tokenizer to implement and debug. Every unique character seen
during train() gets a stable integer ID. Anything unseen at inference time
maps to <UNK>.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from praxis.tokenizer.base import SpecialTokens, Tokenizer

# Reserved vocabulary entries — always assigned the lowest IDs in this order.
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

_RESERVED_TOKENS: tuple[str, ...] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


class CharacterTokenizer(Tokenizer):
    """
    Splits text into individual characters and maps each to an integer ID.

    Usage:
        tokenizer = CharacterTokenizer()
        tokenizer.train(["hello", "world"])
        ids = tokenizer.encode_to_ids("hello")
        text = tokenizer.decode(ids)
    """

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._trained = False

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        self._require_trained()
        return len(self._token_to_id)

    @property
    def vocabulary(self) -> dict[str, int]:
        self._require_trained()
        return dict(self._token_to_id)

    @property
    def special_tokens(self) -> SpecialTokens:
        self._require_trained()
        return SpecialTokens(
            pad=self._token_to_id[PAD_TOKEN],
            bos=self._token_to_id[BOS_TOKEN],
            eos=self._token_to_id[EOS_TOKEN],
            unk=self._token_to_id[UNK_TOKEN],
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus: Iterable[str]) -> None:
        """
        Scan the corpus and build a deterministic character vocabulary.

        Special tokens are always assigned IDs 0–3. Every other unique
        character is sorted alphabetically before receiving an ID so the
        same corpus always produces the same mapping on any machine.
        """
        chars: set[str] = set()
        for text in corpus:
            chars.update(text)

        # Never treat reserved strings as regular characters.
        chars -= set(_RESERVED_TOKENS)

        self._token_to_id = {}
        self._id_to_token = {}

        for idx, token in enumerate(_RESERVED_TOKENS):
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token

        next_id = len(_RESERVED_TOKENS)
        for char in sorted(chars):
            self._token_to_id[char] = next_id
            self._id_to_token[next_id] = char
            next_id += 1

        self._trained = True

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def token_to_id(self, token: str) -> int:
        self._require_trained()
        if token in self._token_to_id:
            return self._token_to_id[token]
        return self._token_to_id[UNK_TOKEN]

    def id_to_token(self, token_id: int) -> str:
        self._require_trained()
        if token_id not in self._id_to_token:
            raise KeyError(f"ID {token_id} is not in the vocabulary (size {self.vocab_size}).")
        return self._id_to_token[token_id]

    def tokens_to_ids(self, tokens: Sequence[str]) -> tuple[int, ...]:
        return tuple(self.token_to_id(t) for t in tokens)

    def ids_to_tokens(self, ids: Sequence[int]) -> tuple[str, ...]:
        return tuple(self.id_to_token(i) for i in ids)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Split text into one token per character."""
        return tuple(text)

    def encode_to_tokens(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> tuple[str, ...]:
        tokens = list(self.tokenize(text))
        if add_special_tokens:
            return (BOS_TOKEN, *tokens, EOS_TOKEN)
        return tuple(tokens)

    def encode_to_ids(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> tuple[int, ...]:
        return self.tokens_to_ids(
            self.encode_to_tokens(text, add_special_tokens=add_special_tokens)
        )

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        self._require_trained()
        reserved_ids = {self._token_to_id[t] for t in _RESERVED_TOKENS}

        parts: list[str] = []
        for token_id in ids:
            if skip_special_tokens and token_id in reserved_ids:
                continue
            parts.append(self.id_to_token(token_id))

        return "".join(parts)

    def encode_to_ids_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> tuple[tuple[int, ...], ...]:
        return tuple(
            self.encode_to_ids(text, add_special_tokens=add_special_tokens) for text in texts
        )

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(self, text: str) -> None:
        tokens = self.tokenize(text)
        ids = self.tokens_to_ids(tokens)
        print(f'Input:  "{text}"')
        print(f"Tokens: {list(tokens)}")
        print(f"IDs:    {list(ids)}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "tokenizer_type": "character",
            "vocabulary": self._token_to_id,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        if payload.get("tokenizer_type") != "character":
            raise ValueError(
                f"Expected tokenizer_type 'character', got {payload.get('tokenizer_type')!r}."
            )

        vocabulary: dict[str, int] = payload["vocabulary"]
        tokenizer = cls()
        tokenizer._token_to_id = dict(vocabulary)
        tokenizer._id_to_token = {idx: token for token, idx in vocabulary.items()}
        tokenizer._trained = True
        return tokenizer

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if not self._trained:
            raise RuntimeError(
                "Tokenizer has not been trained yet. Call train(corpus) first."
            )
