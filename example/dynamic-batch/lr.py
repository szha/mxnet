from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon

data_ctx = mx.cpu()
model_ctx = mx.cpu()
mx.random.seed(0)
np.random.seed(0)

batch_size = 5
bulk_size = 3
num_bulks = 2
num_inputs = 2
num_outputs = 1
num_examples = bulk_size * num_bulks * batch_size
num_hidden_layers = 0

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

def data_loader(X, y, batch_size):
    return gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                 batch_size=batch_size, shuffle=False)

def model(num_outputs, num_hidden_layers, prefix='hybridsequential0_'):
    net = gluon.nn.HybridSequential(prefix=prefix)
    with net.name_scope():
        for i in range(num_hidden_layers):
            net.add(gluon.nn.Dense(2, in_units=2))
        net.add(gluon.nn.Dense(1, in_units=2))
    return net

def train(model_ctx, train_iter, epochs, num_examples, network, net_trainer, dbatch=False):
    for e in range(epochs):
        cumulative_loss = 0
        losses = []
        # inner loop
        autograd.set_bulk_size(bulk_size)
        for i, (data, label) in enumerate(train_iter):
            print("iteration", i)
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            """
            The frontend should be sth like the following. If loss.backward is in scope,
            perform batched backward.

            with mx.dbatch.record():
                with autograd.record():
                    output = network(data)
                    loss = square_loss(output, label)
                    losses.append(loss)
                loss.backward()
            """
            with autograd.record(dbatch_mode=dbatch):
                output = network(data)
                # loss = square_loss(output, label)
                loss = output
                loss.backward()
                losses.append(loss)
    	    if (i+1) == bulk_size:
                net_trainer.step(batch_size)
                for l in losses:
                    cumulative_loss += nd.mean(l).asscalar()
        print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))

X = nd.random.normal(shape=(num_examples, num_inputs))
noise = 0.01 * nd.random.normal(shape=(num_examples,))
y = real_fn(X) + noise

train_data = data_loader(X, y, batch_size)
batch_train_data = data_loader(X, y, batch_size)

# net without batching
net = model(num_outputs, num_hidden_layers)
params = net.collect_params()
params.setattr('grad_req', 'add')
params.initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
trainer = gluon.Trainer(params, 'sgd', {'learning_rate': 0.0001})
#print(params)

# net with batching
batch_net = model(num_outputs, num_hidden_layers)
batch_params = batch_net.collect_params()
#print(params_batch)
# fake init
batch_params.initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
names = params.keys()
for name in names:
   batch_params[name].set_data(params[name].data())
batch_trainer = gluon.Trainer(batch_params, 'sgd', {'learning_rate': 0.0001})

square_loss = gluon.loss.L2Loss()

epochs = 1
loss_sequence = []
num_batches = num_examples / batch_size

train(model_ctx, batch_train_data, epochs, num_examples, batch_net, batch_trainer, dbatch=True)
train(model_ctx, train_data, epochs, num_examples, net, trainer)
