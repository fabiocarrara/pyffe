import caffe
import math

class Solver (object):

	def __init__(self, **kwargs):
		self.init_solver_parameter()
		self.params = dict(
			train_epochs = 3,
			val_interval_epochs = 1,
			val_epochs = 1,
			snapshot_interval_epochs = 1,
			display_per_epoch = 50,
			lr_policy = "fixed",
			stepsize_epochs = 1
		)
		self.params.update(kwargs)
		
		# pass args to SolverParameter
		for k,v in self.params.iteritems():
			if hasattr(self.sp, k):
				setattr(self.sp, k, v)
	
	def init_solver_parameter(self):
		self.sp = caffe.proto.caffe_pb2.SolverParameter()
		self.sp.net = "train_val.prototxt"
		#self.sp.solver_type = "SGD"
		
		# critical:
		self.sp.base_lr = 0.01
		self.sp.momentum = 0.9
		
		# speed:
		# self.sp.test_iter = 100
		# self.sp.test_interval = 250
		
		# looks:
		# self.sp.display = 25
		# self.sp.snapshot = 2500
		self.sp.snapshot_prefix = "snapshots/snapshot"
		
		# learning rate policy
		self.sp.lr_policy = "fixed"

		# important, but rare:
		self.sp.gamma = 0.1
		self.sp.weight_decay = 0.0005
		# self.sp.train_net = trainnet_prototxt_path
		# self.sp.test_net = testnet_prototxt_path

		# pretty much never change these.
		# self.sp.max_iter = 100000
		# self.sp.test_initialization = false
		# self.sp.average_loss = 25 # this has to do with the display.
		# self.sp.iter_size = 1 # this is for accumulating gradients
		self.sp.random_seed = 23
		'''
		if (debug):
			self.sp.max_iter = 12
			self.sp.test_iter = 1
			self.sp.test_interval = 4
			self.sp.display = 1
		'''
		
	def set_train_epoch(self, num, batch_size):
		self.sp.max_iter = int( math.ceil( float(self.params['train_epochs']) * num / batch_size ) )
		self.sp.display = int ( max( round( num / (batch_size * float(self.params['display_per_epoch']) ) ), 1 ) )
		self.sp.average_loss = self.sp.display # display averaged loss
		self.sp.snapshot = int( math.ceil( float(self.params['snapshot_interval_epochs']) * num / batch_size ) )
		self.sp.test_interval = int( math.ceil( float(self.params['val_interval_epochs']) * num / batch_size ) )
		
		if self.sp.lr_policy == "step":
			self.sp.stepsize = int( math.ceil( float(self.params['stepsize_epochs']) * num / batch_size ) )
		
	def set_val_epoch(self, num, batch_size):
		self.sp.test_iter.append( int( math.ceil( float(self.params['val_epochs']) * num / batch_size ) ) )

	def disable_val(self):
		self.sp.test_initialization = False

	def to_solver_prototxt(self):
		return str(self.sp)


