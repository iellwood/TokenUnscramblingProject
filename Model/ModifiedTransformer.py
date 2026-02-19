import torch
import LearnedTransformerAttentionProject.Model.InputLayer as EmbeddingLayers
from LearnedTransformerAttentionProject.Model.EncoderDecoderLayer import EncoderDecoderLayer, EncoderDecoderLayer_WithFAHead


class ModifiedTransformer(torch.nn.Module):

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_sequence_length=220,
                 tgt_sequence_length=220,
                 FA_sequence_length=20,
                 use_modified_layer=True,
                 separate_scramble_matrix_for_K_and_V=False,
                 permute_kv=False,
                 normalize_scramble_matrix_columns=False,
                 embed_dim=512,
                 num_heads=8,
                 number_of_layers=6,
                 modified_layer=3,
                 dim_feedforward=2048,
                 dropout_probability=0.1,
                 device=None):

        super().__init__()

        self.src_sequence_length = src_sequence_length
        self.tgt_sequence_length = tgt_sequence_length
        self.num_heads = num_heads
        self.device = device
        self.embed_dim = embed_dim

        # Mask that prevents the decoder from seeing only tokens that it has already attempted to generate
        self.tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_sequence_length, device=device)

        # Convert language tokens into vectors and add positional encoding
        self.input_layer = EmbeddingLayers.InputLayer(src_sequence_length, tgt_sequence_length, embed_dim, src_vocab_size, trg_vocab_size, dropout_probability, device=device)

        layer_list = []
        for i in range(modified_layer - 1):
            layer_list.append(EncoderDecoderLayer(embed_dim, num_heads, self.tgt_mask, dim_feedforward, activation='gelu', device=device))

        if not use_modified_layer:
            # In the control model, simply use a regular encoder/decoder layer
            layer_list.append(EncoderDecoderLayer(embed_dim, num_heads, self.tgt_mask, dim_feedforward, activation='gelu', device=device))
        else:
            # In the test model, simply use a modified decoder
            layer_list.append(
                              EncoderDecoderLayer_WithFAHead(
                                                   FA_sequence_length,
                                                   separate_scramble_matrix_for_K_and_V,
                                                   embed_dim,
                                                   num_heads,
                                                   self.tgt_mask,
                                                   dim_feedforward,
                                                   activation='gelu',
                                                   device=device,
                                                   permute_kv=permute_kv,
                                                   normalize_scramble_matrix_columns=normalize_scramble_matrix_columns,
                                                   )
                              )

        for i in range(modified_layer + 1, number_of_layers):
            layer_list.append(EncoderDecoderLayer(embed_dim, num_heads, self.tgt_mask, dim_feedforward, activation='gelu', device=device))

        self.layers = torch.nn.ModuleList(layer_list)

        # Final layer that maps the embedding dimension to the vocabulary size
        self.linear = torch.nn.Linear(embed_dim, trg_vocab_size)

    def init_weights(self):
        print('Initializing Model Weights.')
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        src, tgt = self.input_layer(src, tgt)  # token IDs are turned into embedded vectors

        for layer in self.layers:
            src, tgt = layer(src, tgt)

        y = self.linear(tgt)  # Map the output of the transformer into a vector the size of the vocabulary
        return torch.log_softmax(y, dim=-1)  # return a logit for each vocabulary word

    def compute_NLL(self, src, tgt, tgt_shifted):
        logits = self.forward(src=src, tgt=tgt_shifted)
        LLs_Correct = 0.9 * torch.gather(logits, 2, tgt[:, :, None])
        LLs_UniformDistribution = 0.1 * torch.mean(logits, 2, keepdim=True)  # Note label smoothing
        LLs_LabelSmoothed = 0.9 * LLs_Correct + LLs_UniformDistribution
        NLL = -torch.mean(torch.sum(LLs_LabelSmoothed, 0))
        return NLL

    def autoregression(self, src):
        torch.no_grad()
        self.train(False)
        tgt = torch.zeros((self.tgt_sequence_length, 1), dtype=torch.int64, device=self.device)
        tgt[0] = 1
        ids = []

        for i in range(self.tgt_sequence_length):
            logits = self.forward(src, tgt)
            id = torch.argmax(logits[i, 0, :])
            ids.append(int(id.detach().cpu().numpy()))
            if i < self.tgt_sequence_length - 1:
                tgt[i + 1] = id

        torch.enable_grad()
        self.train(True)

        return ids





