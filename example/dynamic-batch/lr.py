from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon

data_ctx = mx.cpu()
model_ctx = mx.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
noise = 0.01 * nd.random_normal(shape=(num_examples,))
y = real_fn(X) + noise

batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                      batch_size=batch_size, shuffle=True)


net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(gluon.nn.Dense(3, in_units=2))
    net.add(gluon.nn.Dense(1, in_units=3))
params = net.collect_params()
params.initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

square_loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(params, 'sgd', {'learning_rate': 0.0001})


epochs = 10
loss_sequence = []
num_batches = num_examples / batch_size

for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record(dbatch_mode=True):
            output = net(data)
            # loss = square_loss(output, label)
        # loss.backward()
	output.backward()
	if (i+1) == 5:
	    raise ValueError
            trainer.step(batch_size)
        # cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
    loss_sequence.append(cumulative_loss)
