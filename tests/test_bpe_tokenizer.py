import pytest
from src.tokenization.bpe import BPETokenizer

def test_initial_state():
    tokenizer = BPETokenizer()
    assert len(tokenizer.vocab) == 256
    assert tokenizer.merges == {}

def test_train_basic_corpus():
    tokenizer = BPETokenizer()
    text = "low low lower newest widest"
    # Starting with 256 bytes (0-255). Target vocab 260 means 4 merges.
    tokenizer.train(text=text, vocab_size=260)
    
    assert len(tokenizer.vocab) == 260
    assert len(tokenizer.merges) == 4

def test_roundtrip_identity():
    tokenizer = BPETokenizer()
    corpus = "the quick brown fox jumps over the lazy dog. 1234567890 !@#$%^&*()_+"
    tokenizer.train(text=corpus, vocab_size=270)
    
    encoded = tokenizer.encode(corpus)
    decoded = tokenizer.decode(encoded)
    assert decoded == corpus

def test_unseen_text_roundtrip():
    tokenizer = BPETokenizer()
    training_text = "machine learning and artificial intelligence systems engineering"
    tokenizer.train(text=training_text, vocab_size=275)
    
    # An unseen string should still encode and decode losslessly
    unseen = "software systems and algorithms!"
    encoded = tokenizer.encode(unseen)
    decoded = tokenizer.decode(encoded)
    assert decoded == unseen

def test_empty_string():
    tokenizer = BPETokenizer()
    tokenizer.train(text="some sample data", vocab_size=260)
    
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""

def test_compression_ratio():
    tokenizer = BPETokenizer()
    corpus = "ab " * 100
    tokenizer.train(text=corpus, vocab_size=258)
    
    raw_bytes_len = len(corpus.encode("utf-8"))
    encoded = tokenizer.encode(corpus)
    
    # Because 'ab' and ' ' repeat frequently, token count should be significantly less than raw bytes
    assert len(encoded) < raw_bytes_len

def test_invalid_target_vocab_size():
    tokenizer = BPETokenizer()
    with pytest.raises(ValueError):
        # Target vocab cannot be smaller than base byte vocabulary (256)
        tokenizer.train(text="sample", vocab_size=200)

def test_no_valid_pairs_to_merge():
    tokenizer = BPETokenizer()
    # A single byte or empty string cannot form pairs
    tokenizer.train(text="a", vocab_size=260)
    # Merges should stop gracefully without throwing index error
    assert len(tokenizer.vocab) == 256
