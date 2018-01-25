import caffe
from caffe import layers as L, params as P
from pyffe import Model

class mAlexNet (Model):

	def __init__(self, input_format, **kwargs):
		params = dict(name="mAlexNet")
		params.update(kwargs)
		Model.__init__(self, input_format, **params)

	def body(self):
		n = caffe.NetSpec()
		
		conv_defaults = dict(
			param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
			weight_filler=dict(type="xavier"),
			bias_filler=dict(type="constant", value=1)
			)
			
		fc_defaults = dict(
			param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)],
			weight_filler=dict(type="xavier"),
			bias_filler=dict(type="constant", value=1)
		)
			
		n.conv1 = L.Convolution(bottom="data", num_output=16, kernel_size=11, stride=4, **conv_defaults)
		n.relu1 = L.ReLU(n.conv1, in_place=True)
		n.pool1 = L.Pooling(n.relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
		
		n.conv2 = L.Convolution(n.pool1, num_output=20, kernel_size=5, stride=1, **conv_defaults)
		n.relu2 = L.ReLU(n.conv2, in_place=True)
		n.pool2 = L.Pooling(n.relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)

		n.conv3 = L.Convolution(n.pool2, num_output=30, kernel_size=3, stride=1, **conv_defaults)
		n.relu3 = L.ReLU(n.conv3, in_place=True)
		n.pool3 = L.Pooling(n.relu3, pool=P.Pooling.MAX, kernel_size=3, stride=2)

		n.fc4 = L.InnerProduct(n.pool3, num_output=48, **fc_defaults)
		n.relu4 = L.ReLU(n.fc4, in_place=True)
		
		n.fc5 = L.InnerProduct(n.relu4, num_output=self.params['num_output'], **fc_defaults)
		
		return n.to_proto(), "fc5"
		
		
if __name__ == "__main__":
	mAlexNet().cmd_tool()
	
