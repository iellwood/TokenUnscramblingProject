import torch
import LearnedTransformerAttentionProject.Model.InputLayer as EmbeddingLayers


class VanillaTransformer(torch.nn.Module):

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_sequence_length=5000,
                 tgt_sequence_length=5000,
                 embed_dim=512,
                 num_heads=8,
                 number_of_layers=6,
                 dim_feedforward=2048,
                 dropout_probability=0.1,
                 device=None):

        super().__init__()

        self.src_sequence_length = src_sequence_length
        self.tgt_sequence_length = tgt_sequence_length
        self.num_heads = num_heads
        self.device = device
        self.embed_dim = embed_dim

        # Convert language tokens into vectors and add positional encoding
        self.input_layer = EmbeddingLayers.InputLayer(src_sequence_length, tgt_sequence_length, embed_dim, src_vocab_size, trg_vocab_size, dropout_probability, device=device)

        self.transformer = torch.nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=number_of_layers,
            num_decoder_layers=number_of_layers,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            dropout=dropout_probability,
        )

        # Final layer that maps the embedding dimension to the vocabulary size
        self.linear = torch.nn.Linear(embed_dim, trg_vocab_size)

        # Mask that prevents the decoder from seeing only tokens that it has already attempted to generate
        self.tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_sequence_length)

    def init_weights(self):
        print('Initializing Model Weights.')
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        src, tgt = self.input_layer(src, tgt)  # token IDs are turned into embedded vectors
        y = self.transformer(src=src, tgt=tgt, tgt_mask=self.tgt_mask)  # apply the transformer
        y = self.linear(y)  # Map the output of the transformer into a vector the size of the vocabulary
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




