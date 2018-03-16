import mxnet as mx
import mxnet.ndarray as nd

x = nd.ones((3, 4))
y = nd.ones((3, 4))

res = []
for i in range(10):
    res.append(x + y)

print(res)
