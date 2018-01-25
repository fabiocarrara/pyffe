import pyffe
from pyffe import Model

import caffe
from caffe import layers as L


def conv_bn(bottom, nout, ks=3, stride=1, pad=0, learn=True):
    if learn:
        param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]

    if isinstance(bottom, str):
        conv = L.Convolution(bottom=bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, param=param, weight_filler=dict(type='msra'),
                             bias_filler=dict(type='constant'))
    else:
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, param=param, weight_filler=dict(type='msra'),
                             bias_filler=dict(type='constant'))

    bn = L.BatchNorm(conv)
    # lrn = L.LRN(bn)
    return conv, bn


def residual_standard_unit(n, nout, s, newdepth=False):
    """
    This creates the 'standard unit' shown on the left side of Figure 5.
    """
    bottom = n.__dict__['tops'].keys()[-1]  # find the last layer in netspec
    stride = 2 if newdepth else 1

    n[s + 'conv1'], n[s + 'bn1'] = conv_bn(n[bottom], ks=3, stride=stride, nout=nout, pad=1)
    n[s + 'relu1'] = L.ReLU(n[s + 'bn1'], in_place=True)
    n[s + 'conv2'], n[s + 'bn2'] = conv_bn(n[s + 'relu1'], ks=3, stride=1, nout=nout, pad=1)

    if newdepth:
        n[s + 'conv_expand'], n[s + 'bn_expand'] = conv_bn(n[bottom], ks=1, stride=2, nout=nout,
                                                                                pad=0)
        n[s + 'sum'] = L.Eltwise(n[s + 'bn2'], n[s + 'bn_expand'])
    else:
        n[s + 'sum'] = L.Eltwise(n[s + 'bn2'], n[bottom])

    n[s + 'relu2'] = L.ReLU(n[s + 'sum'], in_place=True)


def residual_bottleneck_unit(n, nout, s, newdepth=False):
    """
    This creates the 'standard unit' shown on the left side of Figure 5.
    """

    bottom = n.__dict__['tops'].keys()[-1]  # find the last layer in netspec
    stride = 2 if newdepth and nout > 64 else 1

    n[s + 'conv1'], n[s + 'bn1'] = conv_bn(n[bottom], ks=1, stride=stride, nout=nout, pad=0)
    n[s + 'relu1'] = L.ReLU(n[s + 'bn1'], in_place=True)
    n[s + 'conv2'], n[s + 'bn2'] = conv_bn(n[s + 'relu1'], ks=3, stride=1, nout=nout, pad=1)
    n[s + 'relu2'] = L.ReLU(n[s + 'bn2'], in_place=True)
    n[s + 'conv3'], n[s + 'bn3'] = conv_bn(n[s + 'relu2'], ks=1, stride=1, nout=nout * 4, pad=0)

    if newdepth:
        n[s + 'conv_expand'], n[s + 'bn_expand'] = conv_bn(n[bottom], ks=1, stride=stride,
                                                                                nout=nout * 4, pad=0)
        n[s + 'sum'] = L.Eltwise(n[s + 'bn3'], n[s + 'bn_expand'])
    else:
        n[s + 'sum'] = L.Eltwise(n[s + 'bn3'], n[bottom])

    n[s + 'relu3'] = L.ReLU(n[s + 'sum'], in_place=True)


class ResNet(Model):
    def __init__(self, input_format, **kwargs):
        params = dict(name='ResNet', num_output=1000, layers=18)  # defaults
        params.update(kwargs)
        params['name'] += '-' + str(params['layers'])

        Model.__init__(self, input_format, **params)

    def strain_head(self, subset):
        n = caffe.NetSpec()
        # train
        image_data_param = dict(
            source=subset.get_list_absolute_path(),
            batch_size=self.batch_sizes[0],
            crop_size=self.infmt.crop_size,
            im_mean=self.infmt.mean_pixel,
            root_folder=subset.root_folder
        )

        n.data, n.label = L.Python(module='AsyncImageDataLayer', layer='AsyncImageDataLayer',
                                   ntop=2, param_str=str(image_data_param))
        net = n.to_proto()

        net.name = self.name
        return net

#    def val_tail(self, last_top, stage=None):
#        n = caffe.NetSpec()
#        n.loss = L.SoftmaxWithLoss(bottom=[last_top, "label"])
#        n.accuracy = L.Accuracy(bottom=[last_top, "label"])  # , include=dict(phase=caffe.TEST))
#        return n.to_proto()

    def body(self):

        # figure out network structure
        net_defs = {
            18: ([2, 2, 2, 2], 'standard'),
            34: ([3, 4, 6, 3], 'standard'),
            50: ([3, 4, 6, 3], 'bottleneck'),
            101: ([3, 4, 23, 3], 'bottleneck'),
            152: ([3, 8, 36, 3], 'bottleneck'),
        }

        assert self.layers in net_defs.keys(), 'net of depth:{} not defined'.format(self.layers)

        nunits_list, unit_type = net_defs[
            self.layers]  # nunits_list a list of integers indicating the number of layers in each depth.
        nouts = [64, 128, 256, 512]  # same for all nets

        n = caffe.NetSpec()
        # n.name = 'ResNet-' + str(self.layers)

        # setup the first couple of layers
        n.conv1, n.bn1 = conv_bn('data', ks=11, stride=4, nout=nouts[0], pad=0)
        n.pool1 = L.Pooling(n.bn1, stride=2, kernel_size=3)
        n.relu1 = L.ReLU(n.pool1, in_place=True)

        # make the convolutional body
        for nout, nunits in zip(nouts, nunits_list):  # for each depth and nunits
            for unit in range(1, nunits + 1):  # for each unit. Enumerate from 1.
                s = str(nout) + '_' + str(unit) + '_'  # layer name prefix
                if unit_type == 'standard':
                    residual_standard_unit(n, nout, s, newdepth=unit is 1 and nout > nouts[0])
                else:
                    residual_bottleneck_unit(n, nout, s, newdepth=unit is 1)

        # add the end layers
        n.global_pool = L.Pooling(n.__dict__['tops'][n.__dict__['tops'].keys()[-1]],
                                  pooling_param=dict(pool=1, global_pooling=True))
                                  
        n.score = L.InnerProduct(n.global_pool, num_output=self.params['num_output'])
        	#, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

        return n.to_proto(), 'score'


# test it!
if __name__ == '__main__':
    input_format = pyffe.InputFormat(
        new_width=256,
        new_height=256,
        crop_size=224,
        scale=1. / 256,
        mirror=True
    )
    print ResNet(input_format, num_output=2, layers=152).to_deploy_prototxt()
