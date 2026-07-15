import json

import pytest

from praxis.tokenizer.character import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    CharacterTokenizer,
)


@pytest.fixture
def corpus() -> list[str]:
    return ["hello", "world", "abc"]


@pytest.fixture
def tokenizer(corpus: list[str]) -> CharacterTokenizer:
    tok = CharacterTokenizer()
    tok.train(corpus)
    return tok


class TestTraining:
    def test_requires_train_before_use(self) -> None:
        tok = CharacterTokenizer()
        with pytest.raises(RuntimeError, match="not been trained"):
            _ = tok.vocab_size

    def test_special_tokens_get_lowest_ids(self, tokenizer: CharacterTokenizer) -> None:
        assert tokenizer._token_to_id[PAD_TOKEN] == 0
        assert tokenizer._token_to_id[BOS_TOKEN] == 1
        assert tokenizer._token_to_id[EOS_TOKEN] == 2
        assert tokenizer._token_to_id[UNK_TOKEN] == 3

    def test_vocabulary_is_deterministic(self, corpus: list[str]) -> None:
        a = CharacterTokenizer()
        b = CharacterTokenizer()
        a.train(corpus)
        b.train(corpus)
        assert a.vocabulary == b.vocabulary

    def test_characters_are_sorted_after_special_tokens(
        self, tokenizer: CharacterTokenizer
    ) -> None:
        # From corpus "hello", "world", "abc" -> chars a,b,c,d,e,h,l,o,r,w
        regular = {
            token: idx for token, idx in tokenizer.vocabulary.items() if token not in {
                PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
            }
        }
        assert list(regular.keys()) == sorted(regular.keys())
        assert min(regular.values()) == 4

    def test_empty_corpus_still_has_special_tokens(self) -> None:
        tok = CharacterTokenizer()
        tok.train([])
        assert tok.vocab_size == 4


class TestTokenize:
    def test_splits_into_characters(self, tokenizer: CharacterTokenizer) -> None:
        assert tokenizer.tokenize("hi") == ("h", "i")

    def test_unicode_characters(self, tokenizer: CharacterTokenizer) -> None:
        tok = CharacterTokenizer()
        tok.train(["café"])
        assert tokenizer.tokenize("a") == ("a",)
        assert tok.tokenize("é") == ("é",)


class TestEncodeDecode:
    def test_roundtrip_without_special_tokens(self, tokenizer: CharacterTokenizer) -> None:
        text = "hello"
        ids = tokenizer.encode_to_ids(text, add_special_tokens=False)
        assert tokenizer.decode(ids, skip_special_tokens=True) == text

    def test_roundtrip_with_special_tokens(self, tokenizer: CharacterTokenizer) -> None:
        text = "hello"
        ids = tokenizer.encode_to_ids(text, add_special_tokens=True)
        assert ids[0] == tokenizer.special_tokens.bos
        assert ids[-1] == tokenizer.special_tokens.eos
        assert tokenizer.decode(ids) == text

    def test_empty_string_with_special_tokens(self, tokenizer: CharacterTokenizer) -> None:
        ids = tokenizer.encode_to_ids("", add_special_tokens=True)
        assert ids == (
            tokenizer.special_tokens.bos,
            tokenizer.special_tokens.eos,
        )
        assert tokenizer.decode(ids) == ""

    def test_empty_string_without_special_tokens(self, tokenizer: CharacterTokenizer) -> None:
        ids = tokenizer.encode_to_ids("", add_special_tokens=False)
        assert ids == ()
        assert tokenizer.decode(ids) == ""


class TestOOV:
    def test_unknown_character_maps_to_unk(self, tokenizer: CharacterTokenizer) -> None:
        # 'z' was never in the training corpus
        assert tokenizer.token_to_id("z") == tokenizer.special_tokens.unk

    def test_decode_includes_unk_for_oov_ids(self, tokenizer: CharacterTokenizer) -> None:
        unk_id = tokenizer.special_tokens.unk
        # With skip_special_tokens=False, UNK token string appears in output
        result = tokenizer.decode([unk_id], skip_special_tokens=False)
        assert result == UNK_TOKEN


class TestBatch:
    def test_encode_to_ids_batch(self, tokenizer: CharacterTokenizer) -> None:
        batch = tokenizer.encode_to_ids_batch(["hi", "ab"], add_special_tokens=False)
        assert batch == (
            tokenizer.encode_to_ids("hi", add_special_tokens=False),
            tokenizer.encode_to_ids("ab", add_special_tokens=False),
        )


class TestLookups:
    def test_tokens_to_ids_and_ids_to_tokens(self, tokenizer: CharacterTokenizer) -> None:
        tokens = ("h", "e", "l", "l", "o")
        ids = tokenizer.tokens_to_ids(tokens)
        assert tokenizer.ids_to_tokens(ids) == tokens

    def test_invalid_id_raises(self, tokenizer: CharacterTokenizer) -> None:
        with pytest.raises(KeyError, match="not in the vocabulary"):
            tokenizer.id_to_token(9999)

    def test_len_matches_vocab_size(self, tokenizer: CharacterTokenizer) -> None:
        assert len(tokenizer) == tokenizer.vocab_size


class TestPersistence:
    def test_save_and_load_roundtrip(
        self, tokenizer: CharacterTokenizer, tmp_path
    ) -> None:
        path = tmp_path / "vocab.json"
        tokenizer.save(path)

        loaded = CharacterTokenizer.load(path)
        assert loaded.vocabulary == tokenizer.vocabulary
        assert loaded.encode_to_ids("hello") == tokenizer.encode_to_ids("hello")

    def test_save_writes_valid_json(self, tokenizer: CharacterTokenizer, tmp_path) -> None:
        path = tmp_path / "vocab.json"
        tokenizer.save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["tokenizer_type"] == "character"
        assert PAD_TOKEN in payload["vocabulary"]

    def test_load_rejects_wrong_type(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"tokenizer_type": "bpe", "vocabulary": {}}))
        with pytest.raises(ValueError, match="Expected tokenizer_type"):
            CharacterTokenizer.load(path)


class TestExplain:
    def test_explain_does_not_raise(self, tokenizer: CharacterTokenizer, capsys) -> None:
        tokenizer.explain("hi")
        captured = capsys.readouterr()
        assert 'Input:  "hi"' in captured.out
        assert "Tokens:" in captured.out
        assert "IDs:" in captured.out
