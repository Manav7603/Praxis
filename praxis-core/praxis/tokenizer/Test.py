import sys
from pathlib import Path

# Running this file directly doesn't add the project root to Python's path.
# This line fixes: python praxis/tokenizer/Test.py
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from praxis.tokenizer.character import CharacterTokenizer

tokenizer = CharacterTokenizer()
tokenizer.train(["hello world", "playing"])

tokenizer.explain("playing")
print(tokenizer.vocabulary)
# Input:  "playing"
# Tokens: ['p', 'l', 'a', 'y', 'i', 'n', 'g']
# IDs:    [12, 10, 4, 16, 8, 11, 7]

ids = tokenizer.encode_to_ids("hello", add_special_tokens=True)
print(ids)
print(tokenizer.decode(ids))
print(tokenizer.decode(ids, skip_special_tokens=False))
assert tokenizer.decode(ids) == "hello"
assert tokenizer.decode(ids, skip_special_tokens=False) == "<BOS>hello<EOS>"