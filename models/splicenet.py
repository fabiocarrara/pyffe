import caffe
from caffe import layers as L, params as P
from pyffe import Model


def conv_bn_relu(bottom, nout, conv_defaults):
    if isinstance(bottom, str):
        conv = L.Convolution(bottom=bottom, num_output=nout, **conv_defaults)
    else:
        conv = L.Convolution(bottom, num_output=nout, **conv_defaults)

    bn = L.BatchNorm(conv, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
    relu = L.ReLU(bn, in_place=True)
    return conv, bn, relu


def conv_relu(bottom, nout, conv_defaults):
    if isinstance(bottom, str):
        conv = L.Convolution(bottom=bottom, num_output=nout, **conv_defaults)
    else:
        conv = L.Convolution(bottom, num_output=nout, **conv_defaults)

    relu = L.ReLU(conv, in_place=True)
    return conv, relu


class SpliceNet(Model):
    def __init__(self, input_format, **kwargs):
        params = dict(name="SpliceNet", num_output=2)
        params.update(kwargs)
        Model.__init__(self, input_format, **params)

    def body_bn(self, bottom_name='data'):
        n = caffe.NetSpec()

        conv_defaults = dict(
            kernel_size=3,
            stride=1, pad=1,
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            weight_filler=dict(type="xavier", std=0.01),
        )

        fc_defaults = dict(
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            weight_filler=dict(type="xavier", std=0.005),
            bias_filler=dict(type="constant", value=0.1)
        )

        n.conv1_1, n.bn1_1, n.relu1_1 = conv_bn_relu(bottom_name, 64, conv_defaults)
        n.conv1_2, n.bn1_2, n.relu1_2 = conv_bn_relu(n.relu1_1, 64, conv_defaults)
        n.pool1 = L.Pooling(n.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv2_1, n.bn2_1, n.relu2_1 = conv_bn_relu(n.pool1, 128, conv_defaults)
        n.conv2_2, n.bn2_2, n.relu2_2 = conv_bn_relu(n.relu2_1, 128, conv_defaults)
        n.pool2 = L.Pooling(n.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv3_1, n.bn3_1, n.relu3_1 = conv_bn_relu(n.pool2, 256, conv_defaults)
        n.conv3_2, n.bn3_2, n.relu3_2 = conv_bn_relu(n.relu3_1, 256, conv_defaults)
        n.pool3 = L.Pooling(n.relu3_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.fc4 = L.InnerProduct(n.pool3, num_output=1024, **fc_defaults)
        n.relu4 = L.ReLU(n.fc4, in_place=True)
        n.fc5 = L.InnerProduct(n.relu4, num_output=1024, **fc_defaults)
        n.relu5 = L.ReLU(n.fc5, in_place=True)

        n.fc6 = L.InnerProduct(n.relu5, num_output=self.params['num_output'],
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type="xavier", std=0.01),
                               bias_filler=dict(type="constant", value=0)
                               )

        return n.to_proto(), "fc6"

    def body(self, bottom_name='data'):
        n = caffe.NetSpec()

        conv_defaults = dict(
            kernel_size=3,
            stride=1, pad=1,
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            weight_filler=dict(type="xavier", std=0.01),
        )

        fc_defaults = dict(
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            weight_filler=dict(type="xavier", std=0.005),
            bias_filler=dict(type="constant", value=0.1)
        )

        n.conv1_1, n.relu1_1 = conv_relu(bottom_name, 64, conv_defaults)
        n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, conv_defaults)
        n.pool1 = L.Pooling(n.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, conv_defaults)
        n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, conv_defaults)
        n.pool2 = L.Pooling(n.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, conv_defaults)
        n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, conv_defaults)
        n.pool3 = L.Pooling(n.relu3_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.fc4 = L.InnerProduct(n.pool3, num_output=1024, **fc_defaults)
        n.relu4 = L.ReLU(n.fc4, in_place=True)
        n.drop4 = L.Dropout(n.relu4, dropout_ratio=0.5, in_place=True)

        n.fc5 = L.InnerProduct(n.relu4, num_output=1024, **fc_defaults)
        n.relu5 = L.ReLU(n.fc5, in_place=True)
        n.drop5 = L.Dropout(n.relu5, dropout_ratio=0.5, in_place=True)

        n.fc6 = L.InnerProduct(n.drop5, num_output=self.params['num_output'],
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                               weight_filler=dict(type="xavier", std=0.01),
                               bias_filler=dict(type="constant", value=0)
                               )

        return n.to_proto(), "fc6"
