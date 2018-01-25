class InputFormat (object):

	def __init__(self, **kwargs):
		self.__dict__['params'] = dict(
			new_width	= 256,
			new_height	= 256,
			crop_size	= 224,
			mean_pixel	= None,
			mean_file	= None, # path to mean file
			scale		= None,
			mirror		= True,
			pretrain	= None
		)
		self.params.update(kwargs)

	def __getattr__(self, name):
		if 'params' in self.__dict__ and name in self.__dict__['params']:
			return self.__dict__['params'][name]
		
		raise AttributeError("No attribute called {} is present".format(name))
	
	def __setattr__(self, name, value):
		if 'params' in self.__dict__ and name in self.__dict__['params']:
			self.__dict__['params'][name] = value
