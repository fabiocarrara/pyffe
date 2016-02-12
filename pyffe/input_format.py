class InputFormat (object):

	def __init__(self, **kwargs):
		self.params = dict(
			new_width	= 256,
			new_height	= 256,
			crop_size	= 224,
			mean_pixel	= [127, 127, 127],
			mean_file	= None,
			scale		= None,
			# scale		= 1. / 255, # Not good for AlexNet in Binary Classification!
			mirror		= True
		)
		self.params.update(kwargs)


