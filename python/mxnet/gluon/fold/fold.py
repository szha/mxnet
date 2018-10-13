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

# coding: utf-8
# pylint: disable=
"""fold for dynamic batching."""
__all__ = ['batching', 'is_batching', 'NDArrayFuture', 'create_ndarray_future']

import numpy as np
import ctypes
import collections
import random
import sys

from ... import ndarray
from ...base import check_call, _LIB, py_str
from ...ndarray import NDArray


class NDArrayFutureManager(object):
    def __init__(self):
        self.table = {}

    def instantiate(self, future, arr):
        assert future.key not in self.table
        self.table[future.key] = arr

    def look(self, future):
        assert future.key in self.table
        return self.table[future.key]

_CurrentFutureManager = NDArrayFutureManager()

def get_future_manager():
    return _CurrentFutureManager


class NDArrayFuture(NDArray):
    allowed = ['__init__', 'key', 'instantiated', 'manager',
               'instantiate', 'handle', 'writable']

    def __init__(self):
        NDArray.__init__(self, None, True)
        self.key = random.randint(0, sys.maxint)
        self.instantiated = False
        self.manager = get_future_manager()

    def __del__(self):
        print('delete future')

    def __repr__(self):
        if self.instantiated:
            return NDArray.__repr__(self)
        else:
            return '<NDArrayFuture(uninstantiated)>'


    def instantiate(self, arr):
        assert isinstance(arr, NDArray)
        self.manager.instantiate(self, arr)
        NDArray.__init__(self, arr.handle, True)
        self.instantiated = True

    def __getattribute__(self, attr):
        # print('get attr: {0}'.format(attr))
        if attr in NDArrayFuture.allowed:
            return NDArray.__getattribute__(self, attr)

        if not self.instantiated:
            try:
                arr = self.manager.look(self)
            except KeyError:
                raise ValueError, "attribute '{0}' is not allowed for " \
                    "uninstantiated ndarray future.".format(attr)
            NDArray.__init__(self, arr.handle, True)
            self.instantiated = True
            return NDArray.__getattribute__(self, attr)
        else:
            return NDArray.__getattribute__(self, attr)


def create_ndarray_future(arr=None):
    future = NDArrayFuture()
    if arr is not None:
        future.instantiate(arr)
    return future


def calculate_signature(op_name, args, kwargs):
    return 960504 # MAGIC!


# algorithm part for dynamic batching
class Fold(object):
    def __init__(self):
        self.steps = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.depths = {}
        self.exec_list = []
        self.out_futures = []

    def record(self, op_name, out, args, kwargs):
        # print('op_name: {0}'.format(op_name))
        # print('args: {0}'.format(args))
        # print('kwargs: {0}'.format(kwargs))
        # setup input depth
        for arg in args:
            if arg not in self.depths:
                # NDArray or instantiated NDArrayFuture
                if isinstance(arg, NDArrayFuture):
                    assert arg.instantiated
                self.depths[arg] = 0
        step = max([self.depths[arg] for arg in args]) + 1
        self.depths[out] = step
        op_tuple = (op_name, args, kwargs)
        op_sig = calculate_signature(op_name, args, kwargs)
        self.steps[step][op_sig].append(op_tuple)
        self.exec_list.append(op_tuple)
        self.out_futures.append(out)


    def batch(self):
        for step in range(max_step):
            for op_sig in self.steps[step]:
                # exec concatenate
                old_ops = self.steps[step][op_sig]
                new_inputs, concat_op = concat_inputs(old_ops)
                self.exec_list.append(concat_op)
                outputs, new_op = (old_ops[0][0], new_inputs, old_ops[0][2])
                self.exec_list.append(new_op)
                new_outputs, split_op = split_outputs(outputs)
                self.exec_list.append(split_op)


    def execute(self):
        for (op_name, args, kwargs), future in zip(self.exec_list, self.out_futures):
            op = getattr(ndarray, op_name)
            # print(op)
            # print('args: {0}'.format(args))
            out = op(*args, **kwargs)
            future.instantiate(out)
            # print(future)


    def clear(self):
        pass


# batching scope manager

_CurrentBatchingScope = None

def _current_batching_scope():
    return _CurrentBatchingScope

def _set_current_batching_scope(scope):
    global _CurrentBatchingScope
    _CurrentBatchingScope = scope

def batching(batch_size=64):
    return _BatchingScope(batch_size=batch_size)

def is_batching():
    return (_current_batching_scope() is not None)

class _BatchingScope(object):
    def __init__(self, batch_size):
        self._fold = Fold()
        self._batch_size = batch_size

    def __enter__(self):
        print('enter batching scope')
        if _current_batching_scope() is not None:
            raise ValueError, "nested batching scope is not allowed"
        _set_current_batching_scope(self)

    def __exit__(self, ptype, value, trace):
        print('exit batching scope')
        # real execution
        # compute all NDArrayFuture
        # self._fold.batch()
        self._fold.execute()
        self._fold.clear()
        _set_current_batching_scope(None)

    def record(self, op, out, args, kwargs):
        self._fold.record(op, out, args, kwargs)


def _split_batch(arg, batch_axis, arg_size):
    if isinstance(arg, nd.NDArray):
        return nd.split(arg, arg_size, axis=batch_axis) if arg_size > 1 else (arg,)
    arg, fmt = _flatten(arg)
    if arg_size > 1:
        result = (nd.split(x, arg_size, axis=batch_axis) for x in arg)
    else:
        result = ((x,) for x in arg)
    result = zip(*result)
    out = [_regroup(x, fmt)[0] for x in result]
    return out


def _batch_args(arg_lists, arg_types, values, batch_axis):
    res = []
    arg_size = 1
    for arg, arg_type in zip(arg_lists, arg_types):
        arg_size = len(arg)
        if arg_type[0] == 1 and arg[0].batch:
            out = nd.concat(*(x(values) for x in arg), dim=batch_axis)
        elif arg_type[0] == -1:
            out = nd.concat(*arg, dim=batch_axis)
        else:
            for i in range(2, len(arg)):
                assert arg[i] == arg[0], \
                    "Can not use more than one of no-batch argument, got: %s." % str(arg)
            out = arg[0]
            if arg_type[0] == 1:
                out = out(values)
        res.append(out)
    return tuple(res), arg_size


def _make_op_func(name, func_name):
    """Create a NDArray function from the FunctionHandle."""
    code = """
def {0}(*args, **kwargs):
    batching = _current_batching_scope()
    op_name = sys._getframe().f_code.co_name
    future = create_ndarray_future()
    batching.record(op_name, future, args, kwargs)
    return future
    """.format(func_name)
    doc_str = ""

    if name == "FullyConnected":
        print(code)

    local = {}
    exec(code, None, local)  # pylint: disable=exec-used
    op_function = local[func_name]
    op_function.__name__ = func_name
    op_function.__doc__ = doc_str
    op_function.__module__ = 'mxnet.gluon.fold.op'
    return op_function

def _init_ops():
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.MXListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        op_names.append(py_str(plist[i]))

    for name in op_names:
        # print(name)
        func_name = name
        function = _make_op_func(name, func_name)
        cur_module = sys.modules['mxnet.gluon.fold.op']
        function.__module__ = cur_module
        setattr(cur_module, function.__name__, function)
        cur_module.__all__.append(function.__name__)

_init_ops()
