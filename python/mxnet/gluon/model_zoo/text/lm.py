# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Language models."""

from .base import _TextSeq2SeqModel, ExtendedSequential, TransformerBlock
from .base import get_rnn_layer, apply_weight_drop
from ... import nn
from .... import init


class AWDLSTM(_TextSeq2SeqModel):
    """AWD language model."""
    def __init__(self, mode, vocab, embed_dim, hidden_dim, num_layers,
                 dropout=0.5, drop_h=0.5, drop_i=0.5, weight_drop=0,
                 tie_weights=False, **kwargs):
        super(AWDLSTM, self).__init__(vocab, vocab, **kwargs)
        self._mode = mode
        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self._drop_h = drop_h
        self._drop_i = drop_i
<<<<<<< HEAD
=======
        self._drop_e = drop_e
>>>>>>> 42e70287535151781bcda4e7c26d47face1685ab
        self._weight_drop = weight_drop
        self._tie_weights = tie_weights
        self.embedding = self._get_embedding()
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding_block = nn.Embedding(len(self._src_vocab), self._embed_dim,
                                           weight_initializer=init.Uniform(0.1))
<<<<<<< HEAD
=======
            if self._drop_e:
                apply_weight_drop(embedding_block, 'weight', self._drop_e, axes=(1,))
>>>>>>> 42e70287535151781bcda4e7c26d47face1685ab
            embedding.add(embedding_block)
            if self._drop_i:
                embedding.add(nn.Dropout(self._drop_i, axes=(0,)))
        return embedding

    def _get_encoder(self):
        encoder = ExtendedSequential()
        with encoder.name_scope():
            for l in range(self._num_layers):
                encoder.add(get_rnn_layer(self._mode, 1, self._embed_dim if l == 0 else
                                          self._hidden_dim, self._hidden_dim if
                                          l != self._num_layers - 1 or not self._tie_weights
                                          else self._embed_dim, 0, self._weight_drop))
                if self._drop_h:
                    encoder.add(TransformerBlock(nn.Dropout(self._drop_h, axes=(0,)), None))
        return encoder

    def _get_decoder(self):
        vocab_size = len(self._tgt_vocab)
        if self._tie_weights:
<<<<<<< HEAD
            output = nn.Dense(vocab_size, flatten=False, in_units = self._embed_dim, params=self.embedding[0].params)
=======
            output = nn.Dense(vocab_size, flatten=False, params=self.embedding.params)
>>>>>>> 42e70287535151781bcda4e7c26d47face1685ab
        else:
            output = nn.Dense(vocab_size, flatten=False)
        return output

    def begin_state(self, *args, **kwargs):
        return self.encoder[0].begin_state(*args, **kwargs)

class RNNModel(_TextSeq2SeqModel):
    """Simple RNN language model."""
    def __init__(self, mode, vocab, embed_dim, hidden_dim,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        super(RNNModel, self).__init__(vocab, vocab, **kwargs)
        self._mode = mode
        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self._tie_weights = tie_weights
        self.embedding = self._get_embedding()
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _get_embedding(self):
<<<<<<< HEAD
#         embedding = nn.HybridSequential()
#         with embedding.name_scope():
#             embedding.add(nn.Embedding(len(self._src_vocab), self._embed_dim,
#                                        weight_initializer=init.Uniform(0.1)))
#             if self._dropout:
#                 embedding.add(nn.Dropout(self._dropout))
        embedding = nn.Embedding(len(self._src_vocab), self._embed_dim,
                                       weight_initializer=init.Uniform(0.1))
=======
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(nn.Embedding(len(self._src_vocab), self._embed_dim,
                                       weight_initializer=init.Uniform(0.1)))
            if self._dropout:
                embedding.add(nn.Dropout(self._dropout))
>>>>>>> 42e70287535151781bcda4e7c26d47face1685ab
        return embedding

    def _get_encoder(self):
        encoder = ExtendedSequential()
        with encoder.name_scope():
            for l in range(self._num_layers):
                encoder.add(get_rnn_layer(self._mode, 1, self._embed_dim if l == 0 else
                                          self._hidden_dim, self._hidden_dim if
                                          l != self._num_layers - 1 or not self._tie_weights
                                          else self._embed_dim, 0, 0))

        return encoder

    def _get_decoder(self):
        vocab_size = len(self._tgt_vocab)
        if self._tie_weights:
<<<<<<< HEAD
            output = nn.Dense(vocab_size, flatten=False, in_units = self._embed_dim, params=self.embedding[0].params)
=======
            output = nn.Dense(vocab_size, flatten=False, params=self.embedding[0].params)
>>>>>>> 42e70287535151781bcda4e7c26d47face1685ab
        else:
            output = nn.Dense(vocab_size, flatten=False)
        return output

    def begin_state(self, *args, **kwargs):
        return self.encoder[0].begin_state(*args, **kwargs)
