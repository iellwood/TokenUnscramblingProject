import torch
import numpy as np
import tokenizers

class BatchFactory:

    def __init__(self, src_ids_array_path, tgt_ids_array_path, src_tokenizer_path, tgt_tokenizer_path, device=None):

        self.src_ids = np.load(src_ids_array_path)
        self.tgt_ids = np.load(tgt_ids_array_path)

        self.src_tokenizer = tokenizers.Tokenizer.from_file(src_tokenizer_path)
        self.tgt_tokenizer = tokenizers.Tokenizer.from_file(tgt_tokenizer_path)



        self.device = device

    def get_batch(self, batch_size):
        I = np.random.choice(self.src_ids.shape[1], batch_size, replace=False)

        src = torch.from_numpy(self.src_ids[:, I]).to(self.device)
        tgt = torch.from_numpy(self.tgt_ids[:, I]).to(self.device)

        # shift the tgt over to the right by a start token for autoregression

        tgt_shifted = torch.concatenate([torch.ones(size=(1, batch_size), device=self.device, dtype=torch.int64), tgt], dim=0)
        tgt = torch.concatenate([tgt, torch.zeros(size=(1, batch_size), device=self.device, dtype=torch.int64)], dim=0)

        return src, tgt, tgt_shifted

    def get_src_max_sequence_length(self):
        return self.src_ids.shape[0]

    def get_tgt_max_sequence_length(self):
        return self.tgt_ids.shape[0]

    def tokenize_src(self, src):
        src = [int(id) for id in src]
        if 0 in src:
            i = src.index(0)
            src = src[:i]

        src = [i - 2 for i in src]

        return self.src_tokenizer.decode(src)

    def tokenize_tgt(self, tgt):
        tgt = [int(id) for id in tgt]

        if 0 in tgt:
            i = tgt.index(0)
            tgt = tgt[:i]

        tgt = [i - 2 for i in tgt]

        return self.tgt_tokenizer.decode(tgt)

