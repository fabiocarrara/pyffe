import caffe
from caffe import layers as L, params as P
from pyffe import Model

class AlexNet (Model):

	def __init__(self, input_format, **kwargs):
		params = dict(name="AlexNet", num_output=1000)
		params.update(kwargs)
		Model.__init__(self, input_format, **params)

	def body(self, bottom_name='data'):
		n = caffe.NetSpec()
		
		conv_defaults = dict(
			param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
			weight_filler=dict(type="xavier", std=0.01),
		)
		
		lrn_defaults = dict(
			local_size=5,
			alpha=0.0001,
		    beta=0.75
		)
			
		fc_defaults = dict(
			param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
			weight_filler=dict(type="xavier", std=0.005),
			bias_filler=dict(type="constant", value=0.1)
		)
			
		n.conv1 = L.Convolution(bottom=bottom_name, num_output=96, kernel_size=11, stride=4, bias_filler=dict(type="constant", value=0), **conv_defaults)
		n.relu1 = L.ReLU(n.conv1, in_place=True)
		n.norm1 = L.LRN(n.relu1, **lrn_defaults)
		n.pool1 = L.Pooling(n.norm1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
		
		n.conv2 = L.Convolution(n.pool1, num_output=256, kernel_size=5, stride=1, pad=2, group=2, bias_filler=dict(type="constant", value=0.1), **conv_defaults)
		n.relu2 = L.ReLU(n.conv2, in_place=True)
		n.norm2 = L.LRN(n.relu2, **lrn_defaults)
		n.pool2 = L.Pooling(n.norm2, pool=P.Pooling.MAX, kernel_size=3, stride=2)

		n.conv3 = L.Convolution(n.pool2, num_output=384, kernel_size=3, stride=1, pad=1, bias_filler=dict(type="constant", value=0), **conv_defaults)
		n.relu3 = L.ReLU(n.conv3, in_place=True)

		n.conv4 = L.Convolution(n.relu3, num_output=384, kernel_size=3, stride=1, pad=1, group=2, bias_filler=dict(type="constant", value=0.1), **conv_defaults)
		n.relu4 = L.ReLU(n.conv4, in_place=True)
		
		n.conv5 = L.Convolution(n.relu4, num_output=256, kernel_size=3, stride=1, pad=1, group=2, bias_filler=dict(type="constant", value=0.1), **conv_defaults)
		n.relu5 = L.ReLU(n.conv5, in_place=True)
		n.pool5 = L.Pooling(n.relu5, pool=P.Pooling.MAX, kernel_size=3, stride=2)
		
		n.fc6 = L.InnerProduct(n.pool5, num_output=4096, **fc_defaults)
		n.relu6 = L.ReLU(n.fc6, in_place=True)
		n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
		
		n.fc7 = L.InnerProduct(n.drop6, num_output=4096, **fc_defaults)
		n.relu7 = L.ReLU(n.fc7, in_place=True)
		n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
		
		n.fc8 = L.InnerProduct(n.drop7, num_output=self.params['num_output'],
			param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
			weight_filler=dict(type="xavier", std=0.01),
			bias_filler=dict(type="constant", value=0)
		)
		
		return n.to_proto(), "fc8"

