import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.gluon.fold as fd

mx.random.seed(42)

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


class SplitNet(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(SplitNet, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.split(x, num_outputs=2)


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
    net = MLPNet()
    ins = [nd.random_normal(shape=(16, 64)) for i in range(8)]
    check_gluon_net(net, ins)


def test_fold_multi_out():
    net = SplitNet()
    ins = [nd.random_normal(shape=(16, 64)) for i in range(8)]
    check_gluon_net(net, ins)


if __name__ == '__main__':
    test_fold_mlp()
    test_fold_multi_out()
