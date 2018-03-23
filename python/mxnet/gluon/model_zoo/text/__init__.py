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
# pylint: disable=wildcard-import, arguments-differ
r"""Module for pre-defined NLP models.

This module contains definitions for the following model architectures:
-  `AWD`_

You can construct a model with random weights by calling its constructor:

.. code::

    from mxnet.gluon.model_zoo import text
    awd, vocab = text.awd_lstm_lm_1150(vocab)

We provide pre-trained models for all the listed models.
These models can constructed by passing ``pretrained='DatasetName'``:

.. code::

    from mxnet.gluon.model_zoo import text
    awd, vocab = text.awd_lstm_lm_1150(pretrained='wikitext-2')

.. _AWD: https://arxiv.org/abs/1404.5997
"""

from .base import *

from . import lm

from .lm import simple_lstm_lm_650, simple_lstm_lm_1500, awd_lstm_lm_1150

def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    vocab : gluon.text.Vocabulary, default None
        Vocabulary object to be used with the language model.
        Required when not loading from pretrained models.
    pretrained : str or None, default None
        The dataset name on which the pretrained model is trained. Options are 'wikitext2'.
        If None, then no pretrained weights are loaded.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    models = {'simple_lstm_lm_650': simple_lstm_lm_650,
              'simple_lstm_lm_1500': simple_lstm_lm_1500,
              'awd_lstm_lm_1150': awd_lstm_lm_1150}
    name = name.lower()
    if name not in models:
        raise ValueError(
            'Model %s is not supported. Available options are\n\t%s'%(
                name, '\n\t'.join(sorted(models.keys()))))
    return models[name](**kwargs)
