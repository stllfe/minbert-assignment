import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor

from base_bert import BertConfig, BertPreTrainedModel
from utils import get_extended_attention_mask


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()

        self.num_attention_heads = config.num_attention_heads  # base: 12
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads
        )  # base: 768 / 12 = 64
        self.all_head_size = (
            self.num_attention_heads * self.attention_head_size
        )  # base: 12 * 64 = 768

        # initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # this dropout is applied to normalized attention scores following the original implementation of transformer
        # although it is a bit unusual, we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer: nn.Linear) -> Tensor:
        # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # next, we need to produce multiple heads for the proj
        # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
        proj = proj.view(
            bs, seq_len, self.num_attention_heads, self.attention_head_size
        )
        # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
        proj = proj.transpose(1, 2)
        return proj

    def attention(
        self, key: Tensor, query: Tensor, value: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """
        key: [bs, num_attention_heads, seq_len, attention_head_size]
        query: [bs, num_attention_heads, seq_len, attention_head_size]
        value: [bs, num_attention_heads, seq_len, attention_head_size]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
        # attention scores are calculated by multiply query and key
        # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
        # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
        # before normalizing the scores, use the attention mask to mask out the padding token scores
        # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number
        bs, seq_len = key.size(0), key.size(2)
        d_k = key.size(-1)  # same as query vector dimension
        assert d_k == self.attention_head_size
        # equivalent to query [seq_len, attention_head_size] @ key [seq_len, attention_head_size] -> S [seq_len x seq_len]
        # but per batch per attention head in parallel, hence S [bs, num_attention_heads, seq_len, seq_len]
        # S = (query @ key.mT) * (d_k**-0.5)
        S = torch.matmul(query, key.transpose(-1, -2)) * (d_k**-0.5)
        # S = torch.masked_fill(S, mask=attention_mask == 0, value=-torch.inf)
        # normalize the scores
        S += attention_mask
        S = torch.softmax(S, dim=-1)
        S = self.dropout(S)

        # multiply the attention scores to the value and get back V'
        # [seq_len, seq_len] @ [seq_len, attention_head_size] -> [seq_len, attention_head_size]
        # but actually per batch per head -> [bs, num_attention_heads, seq_len, attention_head_size]
        attn = S @ value
        # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
        attn = attn.transpose(2, 1)
        output = attn.reshape(bs, seq_len, -1).contiguous()
        return output

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
        # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        # calculate the multi-head attention
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        # multi-head attention
        self.self_attention = BertSelfAttention(config)
        # add-norm
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # another add-norm
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(
        self,
        input: Tensor,
        output: Tensor,
        dense_layer: nn.Module,
        dropout: nn.Module,
        ln_layer: nn.Module,
    ) -> Tensor:
        """
        this function is applied after the multi-head attention layer or the feed forward layer
        input: the input of the previous layer
        output: the output of the previous layer
        dense_layer: used to transform the output
        dropout: the dropout to be applied
        ln_layer: the layer norm to be applied
        """
        # x = dense_layer(output.view(-1, output.size(-1)))
        # x = x.view(*output.shape)
        # TODO SELF: check how dense layers shapes work here, I suspect for 3D inputs it actually flattens the first (batched) dimensions?
        x = dense_layer(output)
        x = dropout(x)
        return ln_layer(input + x)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf
        each block consists of
        1. a multi-head attention layer (BertSelfAttention)
        2. a add-norm that takes the input and output of the multi-head attention layer
        3. a feed forward layer
        4. a add-norm that takes the input and output of the feed forward layer
        """
        # multi-head attention w/ self.self_attention
        attn_out = self.self_attention(hidden_states, attention_mask)

        # add-norm layer
        x = self.add_norm(
            input=hidden_states,
            output=attn_out,
            dense_layer=self.attention_dense,
            dropout=self.attention_dropout,
            ln_layer=self.attention_layer_norm,
        )

        # feed forward
        ffn_out = self.interm_af(self.interm_dense(x))

        # another add-norm layer
        x = self.add_norm(
            input=x,
            output=ffn_out,
            dense_layer=self.out_dense,
            dropout=self.out_dropout,
            ln_layer=self.out_layer_norm,
        )
        return x


class BertModel(BertPreTrainedModel):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.config = config

        # embedding
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.tk_type_embedding = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.embed_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # for [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids: LongTensor) -> Tensor:
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # get word embedding from self.word_embedding
        inputs_embeds = self.word_embedding(input_ids)

        # get position index and position embedding from self.pos_embedding
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)

        # get token type ids, since we are not consider token type, just a placeholder
        tk_type_ids = torch.zeros(
            input_shape, dtype=torch.long, device=input_ids.device
        )
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        # add three embeddings together
        embeds = inputs_embeds + tk_type_embeds + pos_embeds

        # layer norm and dropout
        embeds = self.embed_layer_norm(embeds)
        embeds = self.embed_dropout(embeds)

        return embeds

    def encode(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # get the extended attention mask for self attention
        # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
        # non-padding tokens with 0 and padding tokens with a large negative number
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask, self.dtype
        )

        # pass the hidden states through the encoder layers
        for i, layer_module in enumerate(self.bert_layers):
            # feed the encoding from the last bert_layer to the next
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states

    def forward(self, input_ids: LongTensor, attention_mask: Tensor) -> Tensor:
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_ids=input_ids)

        # feed to a transformer (a stack of BertLayers)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}
