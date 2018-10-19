import numpy as np
import copy
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.fold as fd
from mxnet.test_utils import assert_almost_equal, default_context

mx.random.seed(42)


def check_gluon_net(net, ins):
    net.initialize()
    outs0 = []
    outs1 = []

    for x in ins:
        outs0.append(net(x))

    with fd.batching():
        for x in ins:
            outs1.append(net(x))

    for a, b in zip(outs0, outs1):
        if isinstance(a, (list, tuple)):
            for i in range(len(a)):
                np.testing.assert_array_equal(a[i].asnumpy(), b[i].asnumpy())
        else:
            np.testing.assert_array_equal(a.asnumpy(), b.asnumpy())


def test_fold_mlp():
    class MLPNet(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(MLPNet, self).__init__(**kwargs)
            with self.name_scope():
                self.fc1 = nn.Dense(64)
                self.fc2 = nn.Dense(64)

        def hybrid_forward(self, F, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    net = MLPNet()
    ins = [nd.random_normal(shape=(16, 64)) for i in range(8)]
    check_gluon_net(net, ins)


def test_fold_multi_out():
    class SplitNet(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(SplitNet, self).__init__(**kwargs)

        def hybrid_forward(self, F, x):
            return F.split(x, num_outputs=2)

    net = SplitNet()
    ins = [nd.random_normal(shape=(16, 64)) for i in range(8)]
    check_gluon_net(net, ins)


def test_fold_rnn():
    class TestRNNLayer(gluon.HybridBlock):
        def __init__(self, cell_type, hidden_size, layout, prefix=None, params=None):
            super(TestRNNLayer, self).__init__(prefix=prefix, params=params)
            self.cell = cell_type(hidden_size, prefix='rnn_')
            self.layout = layout
        def hybrid_forward(self, F, inputs, states, valid_length):
            if isinstance(valid_length, list) and len(valid_length) == 0:
                valid_length = None
            return gluon.contrib.rnn.rnn_cell.unroll(self.cell, inputs, states,
                                                     valid_length=valid_length, layout=self.layout)

    def check_unroll(cell_type, num_states, layout):
       batch_size = 20
       input_size = 50
       hidden_size = 30
       seq_len = 10
       if layout == 'TNC':
           rnn_data = mx.nd.normal(loc=0, scale=1, shape=(seq_len, batch_size, input_size))
       elif layout == 'NTC':
           rnn_data = mx.nd.normal(loc=0, scale=1, shape=(batch_size, seq_len, input_size))
       else:
           print("Wrong layout")
           return
       valid_length = mx.nd.round(mx.nd.random.uniform(low=1, high=10, shape=(batch_size)))
       state_shape = (batch_size, hidden_size)
       states = [mx.nd.normal(loc=0, scale=1, shape=state_shape) for i in range(num_states)]
       cell = cell_type(hidden_size, prefix='rnn_')
       cell.initialize(ctx=default_context())
       if layout == 'TNC':
           cell(rnn_data[0], states)
       else:
           cell(rnn_data[:,0,:], states)
       params1 = cell.collect_params()
       orig_params1 = copy.deepcopy(params1)
       trainer = gluon.Trainer(params1, 'sgd', {'learning_rate' : 0.03})
       with mx.autograd.record():
           res1, states1 = cell.unroll(seq_len, rnn_data, states, valid_length=valid_length,
                                       layout=layout, merge_outputs=True)
       res1.backward()
       trainer.step(batch_size)
       configs = [
               lambda layer: None,
               lambda layer: layer.hybridize(),
               lambda layer: layer.hybridize({'inline_limit': 0}),
               lambda layer: layer.hybridize({'static_alloc': True}),
               lambda layer: layer.hybridize({'static_alloc': True, 'static_shape': True}) ]
       # We can't pass None to a hybrid block, but it accepts an empty list.
       # so we use an empty list to represent valid_length if it's None.
       if valid_length is None:
           valid_length = []
       for config in configs:
           layer = TestRNNLayer(cell_type, hidden_size, layout)
           layer.initialize(ctx=default_context())
           config(layer)
           res2, states2 = layer(rnn_data, states, valid_length)
           params2 = layer.collect_params()
           for key, val in orig_params1.items():
               params2[key].set_data(copy.deepcopy(val.data()))
           trainer = gluon.Trainer(params2, 'sgd', {'learning_rate' : 0.03})
           with mx.autograd.record():
               res2, states2 = layer(rnn_data, states, valid_length)
           assert_almost_equal(res1.asnumpy(), res2.asnumpy(), rtol=0.001, atol=0.0001)
           assert len(states1) == len(states2)
           for i in range(len(states1)):
               assert_almost_equal(states1[i].asnumpy(), states2[i].asnumpy(),
                                   rtol=0.001, atol=0.0001)
           res2.backward()
           trainer.step(batch_size)
           for key, val in params1.items():
               weight1 = val.data()
               weight2 = params2[key].data()
               assert_almost_equal(weight1.asnumpy(), weight2.asnumpy(),
                       rtol=0.001, atol=0.0001)

    cell_types = [(gluon.rnn.RNNCell, 1), (gluon.rnn.LSTMCell, 2),
            (gluon.rnn.GRUCell, 1)]
    for cell_type, num_states in cell_types:
        check_unroll(cell_type, num_states, 'TNC')
        check_unroll(cell_type, num_states, 'NTC')



def test_fold_foreach():
    class ForEachNet(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(ForEachNet, self).__init__(**kwargs)
            with self.name_scope():
                self.func = lambda data, states: (data + states[0], states[0] * 2)
                self.states = nd.random.uniform(shape=(10))

        def hybrid_forward(self, F, x):
            out = F.contrib.foreach(self.func, x, self.states)
            return out

    net = ForEachNet()
    ins = [nd.random_normal(shape=(2, 10)) for i in range(8)]
    check_gluon_net(net, ins)

if __name__ == '__main__':
    # test_fold_mlp()
    # test_fold_multi_out()
    # test_fold_rnn()
    test_fold_foreach()
