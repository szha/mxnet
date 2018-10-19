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
from __future__ import absolute_import
__all__ = ['batching', 'is_batching', 'NDArrayFuture', 'create_ndarray_future']

import numpy as np
import ctypes
import random
import sys
from collections import namedtuple, defaultdict

from .. import ndarray
from .. import symbol
from ..base import check_call, _LIB, py_str, _get_op_name_prefix
from ..ndarray import NDArray

# TODO
# - improve ndarray future
# - optimize batch algorithm

# NOTE: future and inputs should be tuple
_OpRecord = namedtuple('OpRecord', ['op_name', 'future', 'inputs', 'attrs'])
OpSig = namedtuple('OpSig', ['op', 'fmt', 'batch_axis'])


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
        # do nothing to avoid double free
        pass

    def __repr__(self):
        if self.instantiated:
            return NDArray.__repr__(self)
        else:
            return '<NDArrayFuture (uninstantiated) @ {0}>'.format(self.key)

    def instantiate(self, arr):
        assert isinstance(arr, NDArray)
        self.manager.instantiate(self, arr)
        NDArray.__init__(self, arr.handle, True)
        self.instantiated = True

    def __getattribute__(self, attr):
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

def _flatten(args):
    if isinstance(args, (list, tuple)):
        flat = []
        fmts = []
        for i in args:
            arg, fmt = _flatten(i)
            flat.extend(arg)
            fmts.append(fmt)
        return tuple(flat), tuple(fmts)
    else:
        return (args,), int(0)

OP_BATCH_AXIS = {}

def register_attr(op_name, fbatch_axis):
    global OP_BATCH_AXIS
    OP_BATCH_AXIS[op_name] = fbatch_axis

register_attr('FullyConnected', lambda x, y: 0)
register_attr('split', lambda x, y: 0)

def infer_batch_axis(op_name, args, kwargs):
    print('infer batch axis of {0}'.format(op_name))
    if op_name in OP_BATCH_AXIS:
        return OP_BATCH_AXIS[op_name](args, kwargs)
    else:
        return None

def calculate_signature(op_name, args, kwargs):
    batch_axis = infer_batch_axis(op_name, args, kwargs)
    flat_args, fmt = _flatten((args, kwargs))
    op_sig = OpSig(op_name, fmt, batch_axis)
    return op_sig

def parse_name(name, mod_name=None):
    prefix = _get_op_name_prefix(name)
    if len(prefix) > 0:
        func_name = name[len(prefix):]
        submodule_name = prefix[1:-1]
    else:
        func_name = name
        submodule_name = ''

    if mod_name is None:
        return func_name
    else:
        assert mod_name in ['ndarray', 'symbol']
        if len(submodule_name) > 0:
            cur_mod_name = "{0}.{1}.{2}".format('mxnet', mod_name, submodule_name)
        else:
            cur_mod_name = "{0}.{1}".format('mxnet', mod_name)
        cur_mod = sys.modules[cur_mod_name]
        return cur_mod, func_name

def get_num_outputs(op_name, args, kwargs):
    mod, func_name = parse_name(op_name, mod_name='symbol')
    fsym = getattr(mod, func_name)
    new_args = []
    def _get_name():
        _get_name.name_cnt += 1
        return 'x{0}'.format(_get_name.name_cnt)
    _get_name.name_cnt = 0
    for arg in args:
        if isinstance(arg, NDArray):
            new_args.append(symbol.var(_get_name()))
        # list of NDArray
        elif isinstance(arg, (list, tuple)) and isinstance(arg[0], NDArray):
            new_arg = [symbol.var(_get_name()) for _ in arg]
            new_args.append(new_arg)
        else:
            new_args.append(arg)
    sym = fsym(*new_args, **kwargs)
    return len(sym.list_outputs())

# algorithm part for dynamic batching
class Fold(object):
    def __init__(self):
        self.max_step = 0
        self.steps = defaultdict(lambda: defaultdict(list))
        self.depths = {}
        self.exec_list = []

    def record(self, op_name, future, inputs, attrs):
        # setup input depth
        for arg in inputs:
            if arg not in self.depths:
                # NDArray or instantiated NDArrayFuture
                if isinstance(arg, NDArrayFuture):
                    assert arg.instantiated
                self.depths[arg] = 0

        step = max([self.depths[arg] for arg in inputs]) + 1
        self.max_step = max(step, self.max_step)
        for out in future:
            self.depths[out] = step
        op_sig = calculate_signature(op_name, inputs, attrs)
        assert isinstance(future, tuple)
        assert isinstance(inputs, tuple)
        self.steps[step][op_sig].append(_OpRecord(op_name, future, inputs, attrs))

    def _concat_inputs(self, inputs, batch_axis):
        assert isinstance(inputs, tuple)
        concat_attrs = {'dim': batch_axis}
        concat_out = create_ndarray_future()
        concat_op = _OpRecord('concat', (concat_out,), inputs, concat_attrs)
        return concat_out, concat_op

    def _split_output(self, new_op_out, futures, concat_arrs, batch_axis):
        split_attrs = {'concat_arrs': concat_arrs, 'futures_list': futures,
            'inputs': (new_op_out,), 'batch_axis': batch_axis}
        # placeholder for split
        split_op = _OpRecord('deferred_split', None, None, split_attrs)
        return split_op

    def batch(self):
        print('max step: {0}'.format(self.max_step))
        for step in range(self.max_step + 1):
            for op_sig in self.steps[step]:
                batch_axis = op_sig.batch_axis
                old_ops = self.steps[step][op_sig]
                op_name = old_ops[0].op_name
                futures_list = [op.future for op in old_ops]
                num_outputs = len(old_ops[0].future)
                # TODO: for now, only batch first input
                inputs = tuple([op.inputs[0] for op in old_ops])
                params = tuple(old_ops[0].inputs[1:])
                attrs = old_ops[0].attrs

                if batch_axis is None:
                    # do not batch
                    self.exec_list += old_ops
                    continue

                concat_out, concat_op = self._concat_inputs(inputs, batch_axis)
                self.exec_list.append(concat_op)

                new_op_in = (concat_out, ) + params
                new_op_out = tuple([create_ndarray_future() for _ in range(num_outputs)])
                new_op = _OpRecord(op_name, new_op_out, new_op_in, attrs)
                self.exec_list.append(new_op)

                deferred_split_op = self._split_output(new_op_out, futures_list, inputs, batch_axis)
                self.exec_list.append(deferred_split_op)

    def execute_record(self, record):
        print('executing %s' % record.op_name)
        mod, func_name = parse_name(record.op_name, 'ndarray')
        op = getattr(mod, func_name)
        out = op(*record.inputs, **record.attrs)
        if isinstance(out, (list, tuple)):
            num_outs = len(out)
            assert num_outs == len(record.future)
            for i in range(num_outs):
                record.future[i].instantiate(out[i])
        else:
            record.future[0].instantiate(out)
        print('done %s' % record.op_name)

    def execute(self):
        for record in self.exec_list:
            # handle deferred_split
            if record.op_name == "deferred_split":
                concat_arrs = record.attrs['concat_arrs']
                futures_list = record.attrs['futures_list']
                inputs = record.attrs['inputs']
                batch_axis = record.attrs['batch_axis']

                indices = [0]
                acc = 0
                for num_batch in [arr.shape[batch_axis] for arr in concat_arrs]:
                    acc += num_batch
                    indices.append(acc)
                split_ops = []
                for i in range(1, len(indices)):
                    split_attrs = {'axis': batch_axis, 'begin': indices[i - 1], 'end': indices[i]}
                    num_outputs = len(futures_list[i - 1])
                    # assert len(futures[i - 1]) == len(inputs[0])
                    for j in range(num_outputs):
                        split_op = _OpRecord('slice_axis', (futures_list[i - 1][j],), (inputs[0][j],), split_attrs)
                        self.execute_record(split_op)
            else:
                self.execute_record(record)

    def clear(self):
        pass


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
    """batching scope manager """
    def __init__(self, batch_size):
        self._fold = Fold()
        self._batch_size = batch_size

    def __enter__(self):
        if _current_batching_scope() is not None:
            raise ValueError, "nested batching scope is not allowed"
        _set_current_batching_scope(self)

    def __exit__(self, ptype, value, trace):
        # real execution happens
        # compute all NDArrayFuture
        self._fold.batch()
        self._fold.execute()
        self._fold.clear()
        _set_current_batching_scope(None)

    def record(self, op, out, args, kwargs):
        self._fold.record(op, out, args, kwargs)
