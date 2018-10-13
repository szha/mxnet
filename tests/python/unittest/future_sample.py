import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.fold as fd


# no allocation happens
future0 = fd.create_ndarray_future()
future1 = future0
print(future0.key)
print(future1.key)
# print(future0.shape)

# ... do something
# real allocation and execution

arr = nd.ones((16, 16))
future0.instantiate(arr)
print(arr)
print(arr.shape)
print(future0)
print(future0.shape)
print(future1.shape)
