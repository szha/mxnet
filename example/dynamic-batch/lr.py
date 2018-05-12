from __future__ import print_function
import mxnet as mx
import numpy as np
import time, argparse
from mxnet import nd, autograd, gluon, batching

data_ctx = mx.cpu()
model_ctx = mx.cpu()

parser = argparse.ArgumentParser(description='Dynamic Batching')
parser.add_argument('--batch-size', type=int, default=256,
                    help='number of samples per batch')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--profile', action='store_true', help='whether to use profiler')

args = parser.parse_args()

batch_size = args.batch_size
seed = args.seed
profile = args.profile
num_batches = 50
num_inputs = 200
num_hidden = 50
num_outputs = 20
num_examples = batch_size * num_batches
num_hidden_layers = 2

mx.random.seed(args.seed)
np.random.seed(args.seed)

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

def data_loader(X, y):
    return gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                 batch_size=1, shuffle=False)

def model(num_inputs, num_outputs, num_hidden, num_hidden_layers, prefix='hybridsequential0_'):
    net = gluon.nn.HybridSequential(prefix=prefix)
    num_in = num_inputs
    with net.name_scope():
        for i in range(num_hidden_layers):
            net.add(gluon.nn.Dense(num_hidden, in_units=num_in))
            num_in = num_hidden
        net.add(gluon.nn.Dense(1, in_units=num_in))
    return net

def train(model_ctx, train_iter, epochs, num_examples, network,
          net_trainer, batch_size, batch_mode=False):
    batching.set_batch_size(batch_size)
    for e in range(epochs):
        cumulative_loss = 0
        losses = []
        # inner loop
        for i, (data, label) in enumerate(train_iter):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            with batching.batch(batch_mode=batch_mode):
                with autograd.record():
                    output = network(data)
                    # loss = square_loss(output, label)
                    loss = output
                    losses.append(loss)
                loss.backward()
    	    if (i+1) == batch_size:
                # temporarily add ignore_stale_grad=True to profile
                # the case when execution is skipped
                net_trainer.step(batch_size, ignore_stale_grad=True)
                for l in losses:
                    cumulative_loss += nd.mean(l).asscalar()
        print("Epoch %s, loss: %.4f " % (e, cumulative_loss / num_examples))

if profile:
    mx.profiler.set_config(profile_all=True, filename='batch.json')
    mx.profiler.set_state('run')

X = nd.random.normal(shape=(num_examples, num_inputs))
noise = 0.01 * nd.random.normal(shape=(num_examples,))
y = real_fn(X) + noise

train_data = data_loader(X, y)
batch_train_data = data_loader(X, y)

# net without batching
net = model(num_inputs, num_outputs, num_hidden, num_hidden_layers)
params = net.collect_params()
params.setattr('grad_req', 'add')
params.initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
trainer = gluon.Trainer(params, 'sgd', {'learning_rate': 0.0001})
#print(params)

# net with batching
batch_net = model(num_inputs, num_outputs, num_hidden, num_hidden_layers)
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

mx.nd.waitall()
t0 = time.time()

# with batching
train(model_ctx, batch_train_data, epochs, num_examples,
      batch_net, batch_trainer, batch_size, batch_mode=True)

mx.nd.waitall()
t1 = time.time()

# without batching
train(model_ctx, train_data, epochs, num_examples, net, trainer, batch_size)

mx.nd.waitall()
t2 = time.time()
print('w/ batching: %.2f sec' % (t1 - t0))
print('w/o batching: %.2f sec' % (t2 - t1))
if profile:
    mx.profiler.set_state('stop')
