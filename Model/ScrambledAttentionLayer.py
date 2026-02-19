import torch
import math


class ScrambledAttentionLayer(torch.nn.Module):
    """
    Implementation of an attention layer in which the keys and values are scrambled.
    """

    def __init__(self, sequence_length, separate_scramble_matrix_for_K_and_V, vec_dim, device, permute_kv=False, normalize_scramble_matrix_columns=False):
        """
        Creates a scrambled attention layer with a single attention head.
        The key/value sequence length must be specified ahead of time and is fixed.

        :param sequence_length: Number of key/value pairs.
        :param separate_scramble_matrix_for_K_and_V: Whether the keys and values are scrambled with separate matrices.
        :param vec_dim: dimension of a single query, key or value.
        :param device: device that is being used for computation, e.g. 'cuda'.
        :param permute_kv: Whether to permute the key value pairs randomly before scrambling them.
        :param normalize_scramble_matrix_columns: Whether the scramble matrices should be L2 normalized.
        """

        super().__init__()

        self.sequence_length = sequence_length
        self.vec_dim = vec_dim
        self.device = device
        self.separate_scramble_matrix_for_K_and_V = separate_scramble_matrix_for_K_and_V
        self.permute_kv = permute_kv
        self.normalize_scramble_matrix_columns = normalize_scramble_matrix_columns

        self.M_K = torch.nn.Parameter(
            torch.normal(0, 1/vec_dim, size=(vec_dim * sequence_length, sequence_length * vec_dim), dtype=torch.float32),
            requires_grad=True
        )

        self.M_V = torch.nn.Parameter(
            torch.normal(0, 1/vec_dim, size=(vec_dim * sequence_length, sequence_length * vec_dim), dtype=torch.float32),
            requires_grad=True
        )

        # if not using separate scramble matrices, zero out M_V and turn off gradients.
        if not self.separate_scramble_matrix_for_K_and_V:
            self.M_V.requires_grad_(False)
            self.M_V.data = self.M_V * 0

        self.last_qkv_input = None
        self.last_input_to_softmax_attention = None

        if self.normalize_scramble_matrix_columns:
            self.normalize_scramble_matrix_data(self.M_K)
            self.normalize_scramble_matrix_data(self.M_V)

    def get_trimmed_sequence(self, x: torch.tensor, end_index, sequence_length):
        if end_index - sequence_length >= 0:
            return x[(end_index - sequence_length):end_index, :, :]
        else:
            remainder = torch.zeros((self.sequence_length - end_index, x.shape[1], x.shape[2]), device=self.device, dtype=torch.float32)
            return torch.concat([remainder, x[:end_index, :, :]], dim=0)

    @staticmethod
    def normalize_scramble_matrix(m, epsilon=1e-8):
        """
        Normalizes dim=0 of the scramble matrix.
        Returns m = m / (m_norm + epsilon)
        :param m: Scramble matrix
        :param epsilon: epsilon: Size of offset of denominator in normalization.
        :return: Normalized Scramble matrix
        """
        m_norm = torch.linalg.vector_norm(m, dim=0, keepdim=True)
        m = m / (m_norm + epsilon)
        return m

    @staticmethod
    def normalize_scramble_matrix_data(m, epsilon=1e-8):
        """
        Normalizes the scramble matrix's data. Not compatible with autograd. This function is used to
        ensure that the parameters of the scramble matrix are initially normalized.
        :param m: Scramble matrix
        :param epsilon: Size of offset of denominator in normalization M -> M/(norm(M) + epsilon)
        """
        m_norm = torch.linalg.vector_norm(m.data, dim=0, keepdim=True)
        m.data = m.data / (m_norm + epsilon)

    def forward(self, q, k, v):
        """
        Computes the output of the FA layer
        :param q: q.shape = [sequence_length, batch_size, vec_dim]
        :param k: k.shape = [sequence_length, batch_size, vec_dim]
        :param v: v.shape = [sequence_length, batch_size, vec_dim]
        :return:
        """

        if self.permute_kv:
            I = torch.randperm(k.shape[0], device=self.device)
            k = k[I, :, :]
            v = v[I, :, :]

        self.last_qkv_input = [q, k, v]

        batch_size = k.shape[1]
        sequence_length = k.shape[0]
        vec_dim = k.shape[2]

        if self.normalize_scramble_matrix_columns:
            M_K = self.normalize_scramble_matrix(self.M_K)
            M_V = self.normalize_scramble_matrix(self.M_V)
        else:
            M_K = self.M_K
            M_V = self.M_V

        k = torch.permute(k, [1, 0, 2])
        k = torch.reshape(k, [k.shape[0], -1])

        k = torch.matmul(k, M_K)
        k = torch.reshape(k, [batch_size, 1, sequence_length, vec_dim])

        v = torch.permute(v, [1, 0, 2])
        v = torch.reshape(v, [k.shape[0], -1])
        if self.separate_scramble_matrix_for_K_and_V:
            v = torch.matmul(v, M_V)
        else:
            v = torch.matmul(v, M_K)
        v = torch.reshape(v, [batch_size, 1, sequence_length, vec_dim])

        q = torch.permute(q, [1, 0, 2])
        q = q[:, None, :, :]

        self.last_input_to_softmax_attention = [q, k, v]

        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
