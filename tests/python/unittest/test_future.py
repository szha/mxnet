import mxnet as mx
import mxnet.ndarray as nd
import mxnet.fold as fd


def test_future_basic():
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

def test_future_add():
    lhs = fd.create_ndarray_future(nd.ones((16, 16)))
    rhs = fd.create_ndarray_future(nd.ones((16, 16)))
    with fd.batching():
        res = lhs + rhs
    return res


if __name__ == '__main__':
    # test_future_basic()
    test_future_add()

