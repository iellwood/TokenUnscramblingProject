import torch
import math

class InputLayer(torch.nn.Module):
    """
    This class is adapted from https://github.com/gordicaleksa/pytorch-original-transformer
    Based on code by Aleksa Gordić in a pytorch implementation of Vaswani et. al. 2017
    """

    def __init__(self, src_sequence_length, tgt_sequence_length, embed_dim, src_vocab_size, tgt_vocab_size, dropout_probability, device):
        super().__init__()

        self.src_embedding = Embedding(src_vocab_size, embed_dim, device=device)
        self.trg_embedding = Embedding(tgt_vocab_size, embed_dim, device=device)

        # Adds positional information to source/target token's embedding vector
        # (otherwise we'd lose the positional information which is important in human languages)
        self.src_pos_embedding = PositionalEncoding(embed_dim, dropout_probability, expected_max_sequence_length=src_sequence_length, device=device)
        self.trg_pos_embedding = PositionalEncoding(embed_dim, dropout_probability, expected_max_sequence_length=tgt_sequence_length, device=device)

    def forward(self, src_token_ids_batch, tgt_token_ids_batch):
        src_token_ids_batch = self.src_embedding(src_token_ids_batch)
        src_token_ids_batch = self.src_pos_embedding(src_token_ids_batch)

        tgt_token_ids_batch = self.trg_embedding(tgt_token_ids_batch)
        tgt_token_ids_batch = self.trg_pos_embedding(tgt_token_ids_batch)

        return src_token_ids_batch, tgt_token_ids_batch


class Embedding(torch.nn.Module):
    """
    This class is adapted from https://github.com/gordicaleksa/pytorch-original-transformer
    Written by Aleksa Gordić in a pytorch implementation of Vaswani et. al. 2017
    """

    def __init__(self, vocab_size, model_dimension, device=None):
        super().__init__()
        self.embeddings_table = torch.nn.Embedding(vocab_size, model_dimension, device=device)
        self.model_dimension = model_dimension

    def forward(self, token_ids_batch):
        assert token_ids_batch.ndim == 2, f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'

        # token_ids_batch has shape (B, S/T), where B - batch size, S/T max src/trg token-sequence length
        # Final shape will be (B, S/T, D) where D is the model dimension, every token id has associated vector
        embeddings = self.embeddings_table(token_ids_batch)

        # (stated in the paper) multiply the embedding weights by the square root of model dimension
        # Page 5, Chapter 3.4 "Embeddings and Softmax"
        return embeddings * math.sqrt(self.model_dimension)


class PositionalEncoding(torch.nn.Module):
    """
    This class is adapted from https://github.com/gordicaleksa/pytorch-original-transformer
    Written by Aleksa Gordić in a pytorch implementation of Vaswani et. al. 2017
    """

    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length=5000, device=None):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_probability)

        # (stated in the paper) Use sine functions whose frequencies form a geometric progression as position encodings,
        # (learning encodings will also work so feel free to change it!). Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions

        positional_encodings_table = positional_encodings_table.to(device)

        # Register buffer because we want to save the positional encodings table inside state_dict even though
        # these are not trainable (not model's parameters) so they otherwise would be excluded from the state_dict
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # embedding_batch's shape = (B, S/T, D), where S/T max src/trg token-sequence length, D - model dimension
        # So here we get (S/T, D) shape which will get broad-casted to (B, S/T, D) when we try and add it to embeddings
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        # (stated in the paper) Applying dropout to the sum of positional encodings and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + positional_encodings)




