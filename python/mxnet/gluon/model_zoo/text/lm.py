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

from .base import get_rnn_layer, apply_weight_drop
from ... import Block, nn
from .... import init, nd


class AWDLSTM(Block):
    """AWD language model."""
    def __init__(self, mode, vocab_size, embed_dim, hidden_dim, num_layers,
                 tie_weights=False, dropout=0.5, weight_drop=0, drop_h=0.5, drop_i=0.5,
                 **kwargs):
        super(AWDLSTM, self).__init__(**kwargs)
        self._mode = mode
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self._drop_h = drop_h
        self._drop_i = drop_i
        self._weight_drop = weight_drop
        self._tie_weights = tie_weights

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding_block = nn.Embedding(self._vocab_size, self._embed_dim,
                                           weight_initializer=init.Uniform(0.1))
            embedding.add(embedding_block)
            if self._drop_i:
                embedding.add(nn.Dropout(self._drop_i, axes=(0,)))
        return embedding

    def _get_encoder(self):
        encoder = nn.Sequential()
        with encoder.name_scope():
            for l in range(self._num_layers):
                encoder.add(get_rnn_layer(self._mode, 1, self._embed_dim if l == 0 else
                                          self._hidden_dim, self._hidden_dim if
                                          l != self._num_layers - 1 or not self._tie_weights
                                          else self._embed_dim, 0, self._weight_drop))
        return encoder

    def _get_decoder(self):
        if self._tie_weights:
            output = nn.Dense(self._vocab_size, flatten=False, params=self.embedding[0].params)
        else:
            output = nn.Dense(self._vocab_size, flatten=False)
        return output

    def begin_state(self, *args, **kwargs):
        return [c.begin_state(*args, **kwargs) for c in self.encoder]

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        encoded = self.embedding(inputs)
        if not begin_state:
            begin_state = self.begin_state()
        out_states = []
        for e, s in zip(self.encoder, begin_state):
            encoded, state = e(encoded, s)
            out_states.append(state)
            if self._drop_h:
                encoded = nd.Dropout(encoded, p=self._drop_h, axes=(0,))
        out = self.decoder(encoded)
        return out, out_states


class SimpleRNNModel(Block):
    """Simple RNN language model."""
    def __init__(self, mode, vocab_size, embed_dim, hidden_dim,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        if tie_weights:
            assert embed_dim == hidden_dim, "Embedding dimension must be equal to " \
                                            "hidden dimension in order to tie weights. " \
                                            "Got: emb: {}, hid: {}.".format(embed_dim, hidden_dim)
        super(RNNModel, self).__init__(**kwargs)
        self._mode = mode
        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self._tie_weights = tie_weights
        self._vocab_size = vocab_size

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(nn.Embedding(self._vocab_size, self._embed_dim,
                                       weight_initializer=init.Uniform(0.1)))
            if self._dropout:
                embedding.add(nn.Dropout(self._dropout))
        return embedding

    def _get_encoder(self):
        return get_rnn_layer(self._mode, self._num_layers, self._embed_dim,
                             self._hidden_dim, self._dropout, 0)

    def _get_decoder(self):
        if self._tie_weights:
            output = nn.Dense(self._vocab_size, flatten=False, params=self.embedding[0].params)
        else:
            output = nn.Dense(self._vocab_size, flatten=False)
        return output

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        embedded_inputs = self.embedding(inputs)
        if not begin_state:
            begin_state = self.begin_state()
        encoded, state = self.encoder(embedded_inputs, begin_state)
        out = self.decoder(encoded)
        return out, state
