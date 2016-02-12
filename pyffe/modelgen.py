import os
import caffe
from caffe import layers as L, NetSpec
from caffe.proto.caffe_pb2 import NetParameter
import argparse

class ModelGen (object):

	def __init__(self, input_format, **kwargs):
		self.infmt = input_format
		self.params = {
			'name': 'unnamed_net',
			'batch_size': 64,
			'channels': 3,
			'train_list': 'train.txt',
			'val_list': 'val.txt',
			'test_list': 'test.txt',
			'train_root_folder': '', 
			'val_root_folder': '',
			'test_root_folder': '',
			'pretrain': None
		}
		self.params.update(kwargs)
	
	# FIXME
	def param_parser(self):
		parser = argparse.ArgumentParser()
		parser.add_argument("--batch_size", type=int, default=self.params['batch_size'], help="Network batch size (default: 64).")
		parser.add_argument("--channels", default=self.params['channels'], help="Number of channels of input images (DEPRECATED, dims[] should be used).")
		parser.add_argument("--new_width", default=self.infmt.params['new_width'], help="Width of the input images (DEPRECATED, dims[] should be used).")
		parser.add_argument("--new_height", default=self.infmt.params['new_height'], help="Height of the input images (DEPRECATED, dims[] should be used).")
		parser.add_argument("--crop_size", default=self.infmt.params['crop_size'], help="Random crop size done in training.")
		parser.add_argument("--mean_pixel", help="Mean pixel .npy filename.")
		parser.add_argument("--scale", default=self.infmt.params['scale'], help="Scale all pixels by this factor.")
		parser.add_argument("--mirror", default=self.infmt.params['mirror'], help="Randomly mirror the training images on the vertical axis.")
		parser.add_argument("--train_list",	default="train.txt", help="Train list file.")
		parser.add_argument("--val_list", default="val.txt", help="Val list file.")
		parser.add_argument("--train_root_folder", help="Root folder for training images.")
		parser.add_argument("--val_root_folder", help="Root foldel for validation images.")

		return parser
		
	def set_max_batch_size(self):
		feas = 1
		l,u = 1, 2048
		print "Optimizing batch size... "
		
		while l <= u:
			c = (l+u)/2
			self.params['batch_size'] = c
			with open("tmp.prototxt", "w") as f:
				f.write(self.to_deploy_prototxt())
			
			ret = os.system("caffe-try-batch-size tmp.prototxt")

			if ret == 0: # feasible
				feas = c
				l = c + 1
			else:
				u = c - 1
		
		os.remove("tmp.prototxt")
		self.params['batch_size'] = feas
		print "Max batch in gpu mem: {}".format(feas)
		return feas
		
	def deploy_head(self):
		net = NetParameter()
		net.name = self.params['name']
		net.input.append("data")
		inshape = net.input_shape.add()
		inshape.dim.append(self.params['batch_size'])
		inshape.dim.append(self.params['channels'])
		inshape.dim.append(self.infmt.params['crop_size'])
		inshape.dim.append(self.infmt.params['crop_size'])
		return net		
		
	def deploy_tail(self, last_top):
		n = NetSpec()
		n.score = L.Softmax(bottom=last_top)
		return n.to_proto()		
		
	def train_val_head(self):
		n = NetSpec()
		# train
		image_data_param = dict(
			source = self.params['train_list'],
			batch_size = self.params['batch_size'],
			new_width = self.infmt.params['new_width'],
			new_height = self.infmt.params['new_height'],
			root_folder = self.params['train_root_folder'],
			rand_skip = self.params['batch_size'],
			shuffle = True
		)
		
		transform_param = dict(
			mirror = self.infmt.params['mirror'],
			crop_size = self.infmt.params['crop_size'],
			# mean_value = self.infmt.params['mean_pixel'],
		)
		
		if self.infmt.params['scale'] is not None:		
			transform_param['scale'] = self.infmt.params['scale']
		
		if self.infmt.params['mean_file'] is not None:
			transform_param['mean_file'] = os.path.basename(self.infmt.params['mean_file'])
		else:
			transform_param['mean_value'] = self.infmt.params['mean_pixel']
		
		n.data, n.label = L.ImageData(ntop=2, image_data_param=image_data_param, transform_param=transform_param, include=dict(phase=caffe.TRAIN))
		net = n.to_proto()
		
		# val
		n = NetSpec()
		image_data_param['source'] = self.params['val_list']
		image_data_param['root_folder'] = self.params['val_root_folder']
		del image_data_param['rand_skip']
		del image_data_param['shuffle']
		transform_param['mirror'] = False
		n.data, n.label = L.ImageData(ntop=2, image_data_param=image_data_param, transform_param=transform_param, include=dict(phase=caffe.TEST))
		
		net.MergeFrom(n.to_proto())
		net.name = self.params['name']
		return net
		

	def train_val_tail(self, last_top):
		n = NetSpec()
		n.loss = L.SoftmaxWithLoss(bottom=[last_top, "label"])
		n.accuracy = L.Accuracy(bottom=[last_top, "label"], include=dict(phase=caffe.TEST))		
		return n.to_proto()
		
	def test_head(self):
		n = NetSpec()
		# test
		image_data_param = dict(
			source = self.params['test_list'],
			batch_size = self.params['batch_size'],
			new_width = self.infmt.params['new_width'],
			new_height = self.infmt.params['new_height'],
			root_folder = self.params['test_root_folder'],
		)
		
		transform_param = dict(
			crop_size = self.infmt.params['crop_size'],
			# mean_value = self.infmt.params['mean_pixel'],
		)
		
		if self.infmt.params['scale'] is not None:		
			transform_param['scale'] = self.infmt.params['scale']
		
		if self.infmt.params['mean_file'] is not None:
			transform_param['mean_file'] = os.path.basename(self.infmt.params['mean_file'])
		else:
			transform_param['mean_value'] = self.infmt.params['mean_pixel']
		
		n.data, n.label = L.ImageData(ntop=2, image_data_param=image_data_param, transform_param=transform_param, include=dict(phase=caffe.TEST))
		
		net = n.to_proto()
		net.name = self.params['name']
		return net
		
	def test_tail(self, last_top):
		n = NetSpec()
		n.accuracy = L.Accuracy(bottom=[last_top, "label"], include=dict(phase=caffe.TEST))
		return n.to_proto()
		
	# abstract method: must return a NetParameter object and last top name
	def body(self):
		raise NotImplementedError()
		
	def to_deploy_prototxt(self):
		net = self.deploy_head()
		tmp_net, last_top = self.body()
		net.MergeFrom(tmp_net)
		tmp_net = self.deploy_tail(last_top)
		net.MergeFrom(tmp_net)
		return str(net)
		
	def to_train_val_prototxt(self):
		net = self.train_val_head()
		tmp_net, last_top = self.body()
		net.MergeFrom(tmp_net)
		tmp_net = self.train_val_tail(last_top)
		net.MergeFrom(tmp_net)
		return str(net)
	
	def to_test_prototxt(self):
		net = self.test_head()
		tmp_net, last_top = self.body()
		net.MergeFrom(tmp_net)
		tmp_net = self.test_tail(last_top)
		net.MergeFrom(tmp_net)
		return str(net)
	
	# FIXME
	def cmd_tool(self):
		parser = argparse.ArgumentParser()
		parser.add_argument(
			"--deploy",
			action="store_true",
			help="Generates the deploy prototxt file"
		)
		args, net_args = parser.parse_known_args()
		
		net_args_parser = self.param_parser()
		net_args = net_args_parser.parse_args(net_args)

		self.params.update(vars(net_args))
		args = vars(args)
		
		str_net = self.to_deploy_prototxt()	if args['deploy'] else self.to_train_val_prototxt()
		print str_net
		
		
	
