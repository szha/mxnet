import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.gluon.fold as fd

mx.random.seed(42)

class Net(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1 = nn.Dense(64)

    def hybrid_forward(self, F, x):
        # x should be NDArrayFuture
        x = self.fc1(x)
        return x


net = Net()
net.initialize()
ins = [nd.random_normal(shape=(16, 64)) for i in range(8)]
outs0 = []
outs1 = []

for x in ins:
    outs0.append(net(x))

with fd.batching():
    for x in ins:
        outs1.append(net(x))

for a, b in zip(outs0, outs1):
    np.testing.assert_array_equal(a.asnumpy(), b.asnumpy())

# print(Net)
# print(len(outs))
# print('outs: {0}'.format(outs))
