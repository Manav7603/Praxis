"""
Tokenizer base interface for Praxis.

What is a tokenizer?
--------------------
Neural networks only understand numbers. A tokenizer is the adapter between
human-readable text and the integer sequences the model actually processes.

    "Hello world"
         |
         |  tokenize() / encode_to_tokens()   split text into units
         v
    ("Hello", " world")
         |
         |  encode_to_ids()                   map units to integer IDs
         v
    (15496, 995)  ──>  embedding layer  ──>  transformer  ──>  (995, 42, ...)
                                                                    |
         |  decode()                        map IDs back to text      |
         v                                                            |
    " world !"  <────────────────────────────────────────────────────┘

Before any encoding happens, trainable tokenizers (BPE, WordPiece, …) must
learn their vocabulary from a corpus via train(). Character-level tokenizers
can implement train() as a no-op or by scanning for unique characters.

Every concrete tokenizer implements this contract so datasets, training, and
inference can swap backends freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SpecialTokens:
    """
    Integer IDs for tokens that carry structure, not meaning.

    These are not "words" — they tell the model *how* to read a sequence.

    pad  — padding; fills shorter sequences in a batch so every row has the
           same length. The model ignores these (via an attention mask).
    bos  — beginning-of-sequence; optional marker that input starts here.
    eos  — end-of-sequence; tells generation to stop; often appended to targets.
    unk  — unknown; used when the tokenizer meets text not in its vocabulary.
    """

    pad: int | None = None
    bos: int | None = None
    eos: int | None = None
    unk: int | None = None


class Tokenizer(ABC):
    """
    Abstract interface every Praxis tokenizer must satisfy.

    Subclasses (e.g. CharacterTokenizer, BPETokenizer) provide the real
    splitting and vocabulary logic. This class only defines *what* the rest
    of the codebase can rely on.
    """

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Total number of tokens in the vocabulary.

        The model's embedding matrix is shaped (vocab_size, d_model) — this
        property tells the model how many rows that matrix needs.
        """
        ...

    @property
    @abstractmethod
    def vocabulary(self) -> dict[str, int]:
        """
        Full token → ID mapping.

        Useful for debugging, visualization, and inspecting what the tokenizer
        actually learned. Read-only from the caller's perspective.
        """
        ...

    @property
    @abstractmethod
    def special_tokens(self) -> SpecialTokens:
        """
        IDs for structural tokens (pad, bos, eos, unk).

        Data loaders use pad_id to align batch lengths; the generator uses
        eos_id to know when to stop sampling.
        """
        ...

    def __len__(self) -> int:
        """Allow len(tokenizer) as a shorthand for vocab_size."""
        return self.vocab_size

    # ------------------------------------------------------------------
    # Training — learn vocabulary from raw text
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, corpus: Iterable[str]) -> None:
        """
        Build or update the vocabulary from a text corpus.

        BPE, WordPiece, and SentencePiece all require this step before
        encoding is possible. CharacterTokenizer may simply collect every
        unique character it sees. Call save() after training to persist.
        """
        ...

    # ------------------------------------------------------------------
    # Vocabulary lookups
    # ------------------------------------------------------------------

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        """
        Map a single string token to its integer ID.

        Raises if the token is not in the vocabulary (unless your subclass
        maps unknowns to unk_id instead).
        """
        ...

    @abstractmethod
    def id_to_token(self, token_id: int) -> str:
        """Map a single integer ID back to its string token."""
        ...

    @abstractmethod
    def tokens_to_ids(self, tokens: Sequence[str]) -> Sequence[int]:
        """
        Batch version of token_to_id.

        Cleaner than looping when you already have a sequence of tokens
        (e.g. from tokenize() or encode_to_tokens()).
        """
        ...

    @abstractmethod
    def ids_to_tokens(self, ids: Sequence[int]) -> Sequence[str]:
        """
        Batch version of id_to_token.

        Cleaner than looping when converting model output before decode().
        """
        ...

    # ------------------------------------------------------------------
    # Core pipeline: text <-> tokens <-> IDs
    # ------------------------------------------------------------------

    @abstractmethod
    def tokenize(self, text: str) -> Sequence[str]:
        """
        Split raw text into string tokens (no integers yet).

        Returns a Sequence (not necessarily a list) so subclasses can return
        tuples, numpy arrays, or custom sequence types later.

        Examples:
          character-level:  "hi"       -> ("h", "i")
          word-level:       "hi there" -> ("hi", " there")
          BPE:              "playing"  -> ("play", "ing")
        """
        ...

    @abstractmethod
    def encode_to_tokens(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> Sequence[str]:
        """
        Text → string tokens, optionally with special tokens included.

        Like tokenize() but may prepend bos / append eos as string tokens.
        Use this when you want to inspect or visualize the token sequence
        before it becomes integers.
        """
        ...

    @abstractmethod
    def encode_to_ids(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> Sequence[int]:
        """
        Text → integer IDs ready for the model.

        This is what the training loop and inference engine call.
        Typically: encode_to_tokens() → tokens_to_ids().
        """
        ...

    @abstractmethod
    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Integer IDs → human-readable text.

        Typically: ids_to_tokens() → join into a string.
        When skip_special_tokens is True, pad/bos/eos/unk are omitted.
        """
        ...

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------

    @abstractmethod
    def encode_to_ids_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> Sequence[Sequence[int]]:
        """
        encode_to_ids() over many strings at once.

        Training data loaders call this on every batch. Subclasses may
        implement a faster path than looping encode_to_ids().
        """
        ...

    # ------------------------------------------------------------------
    # Explain — Praxis-specific debugging / education tool
    # ------------------------------------------------------------------

    @abstractmethod
    def explain(self, text: str) -> None:
        """
        Print a human-readable breakdown of how this text is tokenized.

        Example output for tokenizer.explain("playing"):

            Input:  "playing"
            Tokens: ['p', 'l', 'a', 'y', 'i', 'n', 'g']   (character-level)
            IDs:    [16, 43, 6, 62, 24, 45, 19]

        No other tokenizer library exposes this — it makes the black box
        visible while you are learning and debugging.
        """
        ...

    # ------------------------------------------------------------------
    # Persistence — save trained vocab, reload for inference
    # ------------------------------------------------------------------

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Write vocabulary and config to disk.

        After train(), save so you can reload the exact same vocab when
        resuming training or running inference.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Tokenizer:
        """
        Reconstruct a tokenizer from a path written by save().

        Returns a fully usable instance with the same vocab as when saved.
        """
        ...
