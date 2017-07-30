import mxnet as mx
from mxnet import gluon
from mxnet.gluon.block import Recorder
from mxnet.test_utils import assert_almost_equal

import mxnet.gluon.model_zoo.vision as gmodels
from torchvision import models as pmodels
from torch.autograd import Variable
import torch
import numpy as np
import os
import hashlib

models = {'resnet18_v1': pmodels.resnet18(pretrained=True),
          'resnet34_v1': pmodels.resnet34(pretrained=True),
          'resnet50_v1': pmodels.resnet50(pretrained=True),
          'resnet101_v1': pmodels.resnet101(pretrained=True),
          'resnet152_v1': pmodels.resnet152(pretrained=True),
          'vgg11': pmodels.vgg11(pretrained=True),
          'vgg13': pmodels.vgg13(pretrained=True),
          'vgg16': pmodels.vgg16(pretrained=True),
          'vgg19': pmodels.vgg19(pretrained=True),
          'vgg11_bn': pmodels.vgg11_bn(pretrained=True),
          'vgg13_bn': pmodels.vgg13_bn(pretrained=True),
          'vgg16_bn': pmodels.vgg16_bn(pretrained=True),
          'vgg19_bn': pmodels.vgg19_bn(pretrained=True),
          'alexnet': pmodels.alexnet(pretrained=True),
          'densenet121': pmodels.densenet121(pretrained=True),
          'densenet161': pmodels.densenet161(pretrained=True),
          'densenet169': pmodels.densenet169(pretrained=True),
          'densenet201': pmodels.densenet201(pretrained=True),
          'squeezenet1.0': pmodels.squeezenet1_0(pretrained=True),
          'squeezenet1.1': pmodels.squeezenet1_1(pretrained=True),
          'inceptionv3': pmodels.inception_v3(pretrained=True),
         }

def move_weight(gmodel_name, pmodel, size=(32, 3, 224, 224)):
    if os.path.exists(gmodel_name+'.params'):
        print 'skipped '+gmodel_name
        return
    Recorder.records={}
    gmodel = gmodels.get_model(gmodel_name, classes=1000)
    gmodel.collect_params().initialize(mx.init.Zero(), ctx=mx.cpu(0))
    data = mx.nd.ones(size, ctx=mx.cpu(0))
    gmodel(data)

    pmodel = pmodel.eval()
    pmodel(Variable(torch.ones(size)))

    list_gluon = [i for i in Recorder.records['gluon'] if len(i.items())]
    list_pytorch = [i for i in Recorder.records['torch'] if len(i)]
    problem_list = []

    def torch_param_to_numpy(param):
        if isinstance(param, Variable):
            param = param.data
        return param.numpy()

    def set_gluon_param(param, torch_param):
        if torch_param is not None:
            value = torch_param_to_numpy(torch_param)
            assert param.data().shape == value.shape, str(param)+' has problem. trying shape:'+str(value.shape)
            param.set_data(mx.nd.array(value))

    for g, p in zip(list_gluon, list_pytorch):
        glayer = g
        player = p
        if 'batchnorm' in str(glayer):
            print 'found problem in {} related to batch norm.'.format(glayer.keys())
            new_player = {}
            new_player['beta'] = player['bias']
            new_player['gamma'] = player['weight']
            new_player['running_mean'] = player['running_mean']
            new_player['running_var'] = player['running_var']
            player = new_player
            problem_list.append((g, p))
        if len(glayer.items()) != len(player):
            if len(glayer.items()) < len(player):
                player = {k:v for k,v in player.items() if v is not None}
                if len(player) != len(glayer.items()):
                    raise ValueError(str(glayer)+' has problem even after filtering. ' + 'player: {}'.format(player))
            else:
                raise ValueError(str(glayer)+' has problem. ' + 'player: {}'.format(player))
        for gkey, pkey in zip(sorted(glayer.keys()), sorted(player.keys())):
            set_gluon_param(glayer[gkey], player[pkey])

    def sha1(file_path):
        sha1 = hashlib.sha1()
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)

        return sha1.hexdigest()[:8]

    def random_test():
        c = np.random.uniform(size=size)

        pinput = torch.Tensor(c)
        ginput = mx.nd.array(c, ctx=mx.cpu(0))

        gout = gmodel(ginput).asnumpy()
        pout = pmodel(Variable(pinput)).data.numpy()

        assert_almost_equal(gout, pout, atol=1e-4, rtol=1e-4)
        model_file = gmodel_name+'.params'
        gmodel.save_params(model_file)
        short_sha1 = sha1(model_file)
        basename = '.'.join(model_file.split('.')[:-1]) + '-%s'%short_sha1
        os.rename(model_file, basename+'.params')

    random_test()

for name in sorted(models.keys()):
    model = models[name]
    print 'processing ' +name
    try:
        if 'inception' not in name:
            move_weight(name, model)
        else:
            model.aux_logits=False
            model.transform_input=False
            move_weight(name, model, size=(32, 3, 299, 299))
    except Exception as e:
        print 'failed to copy ' + name
        print 'reason: ' + str(e)
