class InputFormat (object):

	def __init__(self, **kwargs):
		self.__dict__['params'] = dict(
			new_width	= 256,
			new_height	= 256,
			crop_size	= 224,
			mean_pixel	= [127, 127, 127],
			mean_file	= None, # path to mean file
			scale		= None,
			# scale		= 1. / 255, # Not good for AlexNet in Binary Classification!
			mirror		= True,
			pretrain	= None
		)
		self.params.update(kwargs)

	def __getattr__(self, name):
		if name in self.params:
			return self.params[name]
		
		raise AttributeError("No attribute called {} is present".format(name))
	
	def __setattr__(self, name, value):
		if name in self.params:
			self.params[name] = value
