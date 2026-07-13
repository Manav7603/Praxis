Commit - 3:

Why:
A stable tokenizer contract is required before implementing any concrete tokenizer.

What:
- Added abstract Tokenizer base class
- Added SpecialTokens dataclass
- Defined vocabulary, encoding, decoding and persistence APIs
- Documented the tokenizer lifecycle

Impact:
Future tokenizers (Character, BPE, WordPiece, SentencePiece) can share the same interface without changing the training or inference pipeline.