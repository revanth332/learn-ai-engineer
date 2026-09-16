class BPETokenizer:
    """Byte-Pair Encoding (BPE) Tokenizer operating on raw UTF-8 bytes."""

    def __init__(self) -> None:
        # Base vocabulary: mapping from token id (0-255) to single byte
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # Merges: mapping from (token_id1, token_id2) tuple to merged token_id
        self.merges: dict[tuple[int, int], int] = {}

    def train(self, text: str, vocab_size: int) -> None:
        """
        Train the tokenizer on input text until vocab reaches vocab_size.
        
        Args:
            text: Raw input training corpus.
            vocab_size: Target vocabulary size (must be >= 256).
        """
        if vocab_size < 256:
            raise ValueError("Invalid Vocabulary Size")
        n_merges = vocab_size - 256
        token_list = list(text.encode("utf-8"))
        next_token_id = 256
        for _ in range(n_merges):
            mp = {}
            for j in range(len(token_list)-1):
                key = (token_list[j],token_list[j+1])
                mp[key] = mp.get(key,0) + 1
            if not mp:
                return
            freq_pair = max(mp,key=mp.get)
            self.merges[freq_pair] = next_token_id
            self.vocab[next_token_id] = self.vocab[freq_pair[0]] + self.vocab[freq_pair[1]]
            new_token_list = []
            k = 0
            while k < len(token_list):
                if k < (len(token_list)-1) and token_list[k] == freq_pair[0] and token_list[k+1] == freq_pair[1]:
                    new_token_list.append(next_token_id)
                    k += 2
                else:
                    new_token_list.append(token_list[k])
                    k += 1
            next_token_id += 1
            token_list = new_token_list

    def encode(self, text: str) -> list[int]:
        """
        Encode text into a list of token IDs.
        
        Args:
            text: Raw text string.
            
        Returns:
            List of integer token IDs.
        """
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            valid_pairs = [(tokens[i],tokens[i+1]) for i in range(len(tokens)-1) if (tokens[i],tokens[i+1]) in self.merges]
            if not valid_pairs:
                break
            best_valid_pair = min(valid_pairs, key=lambda pair: self.merges[pair])
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == best_valid_pair[0] and tokens[i+1] == best_valid_pair[1]:
                    new_tokens.append(self.merges[best_valid_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.
        
        Args:
            ids: List of integer token IDs.
            
        Returns:
            Reconstructed string.
        """
        byte_chunks = [self.vocab[_id] for _id in ids]
        byte_str = b"".join(byte_chunks)
        return byte_str.decode('utf-8',errors='replace')
        

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.train("low lower lowest",258)
    print(tokenizer.merges)
    print([(key,val.decode('utf-8')) for key,val in tokenizer.vocab.items() if key > 255])
    print(tokenizer.encode("low lower lowest"))
    tokenizer.decode([257, 32, 257, 101, 114, 32, 257, 101, 115, 116])