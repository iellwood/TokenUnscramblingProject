import torch
from LearnedTransformerAttentionProject.Model.ScrambledAttentionLayer import ScrambledAttentionLayer


class EncoderDecoderLayer(torch.nn.Module):

    def __init__(self, d_model, nhead, tgt_mask, dim_feedforward=2048, dropout=0.1, activation=None, device=None):

        super().__init__()

        self.encoder = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, device=device)
        self.decoder = torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, device=device)
        self.tgt_mask = tgt_mask

    def forward(self, src, tgt):

        src = self.encoder(src=src)
        tgt = self.decoder(memory=src, tgt=tgt, tgt_mask=self.tgt_mask)

        return src, tgt


class EncoderDecoderLayer_WithFAHead(torch.nn.Module):

    def __init__(self,
                 FA_sequence_length,
                 separate_scramble_matrix_for_K_and_V,
                 d_model, nhead, tgt_mask, dim_feedforward=2048,
                 dropout=0.1,
                 activation=None,
                 device=None,
                 permute_kv=False,
                 normalize_scramble_matrix_columns=False
                 ):
        super().__init__()

        self.encoder = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, device=device)
        self.decoder = torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, device=device)
        self.vec_dimension = d_model // nhead
        self.FA_sequence_length = FA_sequence_length
        self.FALayer = ScrambledAttentionLayer(FA_sequence_length,
                                               separate_scramble_matrix_for_K_and_V,
                                               self.vec_dimension,
                                               device=device,
                                               permute_kv=permute_kv,
                                               normalize_scramble_matrix_columns=normalize_scramble_matrix_columns
                                               )

        self.tgt_mask = tgt_mask

        self.V_linear = torch.nn.Linear(d_model, self.vec_dimension)
        self.K_linear = torch.nn.Linear(d_model, self.vec_dimension)
        self.Q_linear = torch.nn.Linear(d_model, self.vec_dimension)


    def forward(self, src, tgt):

        x = torch.concatenate([src[:, :, :self.vec_dimension], tgt[:, :, :self.vec_dimension]], dim=0)

        k = self.K_linear(src[:self.FA_sequence_length, :, :])
        v = self.V_linear(src[:self.FA_sequence_length, :, :])
        q = self.Q_linear(tgt)

        tgt_FA = self.FALayer(q, k, v)[:, 0, :, :]
        tgt_FA = torch.permute(tgt_FA, [1, 0, 2])

        src = self.encoder(src=src)
        tgt = self.decoder(memory=src, tgt=tgt, tgt_mask=self.tgt_mask)

        tgt = torch.concatenate([tgt_FA, tgt[:, :, self.vec_dimension:]], dim=2)

        return src, tgt
