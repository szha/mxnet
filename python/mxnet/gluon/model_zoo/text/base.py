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
"""Building blocks and utility for models."""
__all__ = ['StatefulBlock', 'get_rnn_layer', 'get_rnn_cell',
           'RNNCellLayer', 'apply_weight_drop', 'WeightDropParameter']

from ... import Block, HybridBlock, Parameter, contrib, rnn
from .... import nd

class StatefulBlock(Block):
    def __init__(self, **kwargs):
        super(StatefulBlock, self).__init__(**kwargs)

    def begin_state(self, *args, **kwargs):
        raise NotImplementedError()

def apply_weight_drop(block, local_param_name, rate, axes=(),
                      weight_dropout_mode='training'):
    if not rate:
        return

    params = block.collect_params('.*{}'.format(local_param_name))
    for full_param_name, param in params.items():
        dropped_param = WeightDropParameter(param, rate, weight_dropout_mode, axes)
        param_dicts, reg_param_dicts = _find_param(block, full_param_name, local_param_name)
        for param_dict in param_dicts:
            param_dict[full_param_name] = dropped_param
        for reg_param_dict in reg_param_dicts:
            reg_param_dict[local_param_name] = dropped_param
        local_attr = getattr(block, local_param_name)
        if local_attr == param:
            super(Block, block).__setattr__(local_param_name, dropped_param)
        else:
            if isinstance(local_attr, (list, tuple)):
                if isinstance(local_attr, tuple):
                    local_attr = list(local_attr)
                for i, v in enumerate(local_attr):
                    if v == param:
                        local_attr[i] = dropped_param
            elif isinstance(local_attr, dict):
                for k, v in local_attr:
                    if v == param:
                        local_attr[k] = dropped_param
            else:
                continue
            super(Block, block).__setattr__(local_param_name, local_attr)


def _find_param(block, full_param_name, local_param_name):
    param_dict_results = []
    reg_dict_results = []
    params = block.params

    if full_param_name in block.params._params:
        if isinstance(block, HybridBlock) and local_param_name in block._reg_params:
            reg_dict_results.append(block._reg_params)
        while params:
            if full_param_name in params._params:
                param_dict_results.append(params._params)
            if params._shared:
                params = params._shared
            else:
                break

    if block._children:
        for c in block._children:
            pd, rd = _find_param(c, full_param_name, local_param_name)
            param_dict_results.extend(pd)
            reg_dict_results.extend(rd)

    return param_dict_results, reg_dict_results

def get_rnn_cell(mode, num_layers, input_dim, hidden_dim,
                 dropout, weight_dropout,
                 var_drop_in, var_drop_state, var_drop_out):
    """create rnn cell given specs"""
    rnn_cell = rnn.SequentialRNNCell()
    with rnn_cell.name_scope():
        for i in range(num_layers):
            if mode == 'rnn_relu':
                cell = rnn.RNNCell(hidden_dim, 'relu', input_size=input_dim)
            elif mode == 'rnn_tanh':
                cell = rnn.RNNCell(hidden_dim, 'tanh', input_size=input_dim)
            elif mode == 'lstm':
                cell = rnn.LSTMCell(hidden_dim, input_size=input_dim)
            elif mode == 'gru':
                cell = rnn.GRUCell(hidden_dim, input_size=input_dim)
            if var_drop_in + var_drop_state + var_drop_out != 0:
                cell = contrib.rnn.VariationalDropoutCell(cell,
                                                          var_drop_in,
                                                          var_drop_state,
                                                          var_drop_out)

            rnn_cell.add(cell)
            if i != num_layers - 1 and dropout != 0:
                rnn_cell.add(rnn.DropoutCell(dropout))

            if weight_dropout:
                apply_weight_drop(rnn_cell, 'h2h_weight', rate=weight_dropout)

    return rnn_cell


def get_rnn_layer(mode, num_layers, input_dim, hidden_dim, dropout, weight_dropout):
    """create rnn layer given specs"""
    if mode == 'rnn_relu':
        block = rnn.RNN(hidden_dim, 'relu', num_layers, dropout=dropout,
                        input_size=input_dim)
    elif mode == 'rnn_tanh':
        block = rnn.RNN(hidden_dim, num_layers, dropout=dropout,
                        input_size=input_dim)
    elif mode == 'lstm':
        block = rnn.LSTM(hidden_dim, num_layers, dropout=dropout,
                         input_size=input_dim)
    elif mode == 'gru':
        block = rnn.GRU(hidden_dim, num_layers, dropout=dropout,
                        input_size=input_dim)
    if weight_dropout:
        apply_weight_drop(block, 'h2h_weight', rate=weight_dropout)

    return block


class RNNCellLayer(Block):
    """A block that takes an rnn cell and makes it act like rnn layer."""
    def __init__(self, rnn_cell, layout='TNC', **kwargs):
        super(RNNCellLayer, self).__init__(**kwargs)
        self.cell = rnn_cell
        assert layout == 'TNC' or layout == 'NTC', \
            "Invalid layout %s; must be one of ['TNC' or 'NTC']"%layout
        self._layout = layout
        self._axis = layout.find('T')
        self._batch_axis = layout.find('N')

    def forward(self, inputs, states=None): # pylint: disable=arguments-differ
        batch_size = inputs.shape[self._batch_axis]
        skip_states = states is None
        if skip_states:
            states = self.cell.begin_state(batch_size, ctx=inputs.context)
        if isinstance(states, ndarray.NDArray):
            states = [states]
        for state, info in zip(states, self.cell.state_info(batch_size)):
            if state.shape != info['shape']:
                raise ValueError(
                    "Invalid recurrent state shape. Expecting %s, got %s."%(
                        str(info['shape']), str(state.shape)))
        states = sum(zip(*((j for j in i) for i in states)), ())
        outputs, states = self.cell.unroll(
            inputs.shape[self._axis], inputs, states,
            layout=self._layout, merge_outputs=True)

        if skip_states:
            return outputs
        return outputs, states


class WeightDropParameter(Parameter):
    """A Container holding parameters (weights) of Blocks and performs dropout.
    parameter : Parameter
        The parameter which drops out.
    rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
        Dropout is not applied if dropout_rate is 0.
    mode : str, default 'training'
        Whether to only turn on dropout during training or to also turn on for inference.
        Options are 'training' and 'always'.
    axes : tuple of int, default ()
        Axes on which dropout mask is shared.
    """
    def __init__(self, parameter, rate=0.0, mode='training', axes=()):
        p = parameter
        super(WeightDropParameter, self).__init__(
            name=p.name, grad_req=p.grad_req, shape=p._shape, dtype=p.dtype,
            lr_mult=p.lr_mult, wd_mult=p.wd_mult, init=p.init,
            allow_deferred_init=p._allow_deferred_init,
            differentiable=p._differentiable)
        self._rate = rate
        self._mode = mode
        self._axes = axes

    def data(self, ctx=None):
        """Returns a copy of this parameter on one context. Must have been
        initialized on this context before.
        Parameters
        ----------
        ctx : Context
            Desired context.
        Returns
        -------
        NDArray on ctx
        """
        d = self._check_and_get(self._data, ctx)
        if self._rate:
            d = nd.Dropout(d, self._rate, self._mode, self._axes)
        return d

    def __repr__(self):
        s = 'WeightDropParameter {name} (shape={shape}, dtype={dtype}, rate={rate}, mode={mode})'
        return s.format(name=self.name, shape=self.shape, dtype=self.dtype,
                        rate=self._rate, mode=self._mode)
