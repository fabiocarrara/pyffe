#!/usr/bin/env python
#-*- coding:utf-8 -*-

import argparse
import os
import PIL.Image
import numpy as np
import scipy.misc
from google.protobuf import text_format

# os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2

import Queue
import threading
import time

def get_transformer(deploy_file, mean_file=None, mean_pixel=None):
	"""
	Returns an instance of caffe.io.Transformer

	Arguments:
	deploy_file -- path to a .prototxt file

	Keyword arguments:
	mean_pixel -- numpy array with mean pixel in BGR format
	mean_file -- path to a .binaryproto file (optional)
	"""
	network = caffe_pb2.NetParameter()
	with open(deploy_file) as infile:
		text_format.Merge(infile.read(), network)

	if network.input_shape:
		dims = network.input_shape[0].dim
	else:
		dims = network.input_dim[:4]

	t = caffe.io.Transformer(
			inputs = {'data': dims}
			)
	t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

	# color images
	if dims[1] == 3:
		# channel swap
		t.set_channel_swap('data', (2,1,0))

	if mean_file:
		# set mean pixel
		with open(mean_file,'rb') as infile:
			blob = caffe_pb2.BlobProto()
			blob.MergeFromString(infile.read())
			if blob.HasField('shape'):
				blob_dims = blob.shape
				assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
			elif blob.HasField('num') and blob.HasField('channels') and \
					blob.HasField('height') and blob.HasField('width'):
				blob_dims = (blob.num, blob.channels, blob.height, blob.width)
			else:
				raise ValueError('blob does not provide shape or 4d dimensions')
			pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
			t.set_mean('data', pixel)
	
	if mean_pixel:
		t.set_mean('data', np.load(mean_pixel))
	
	return t

def load_and_prepare_image(transformer, path, height, width, mode='RGB'):
	"""
	Load an image from disk

	Returns an np.ndarray (channels x width x height)

	Arguments:
	path -- path to an image on disk
	width -- resize dimension
	height -- resize dimension

	Keyword arguments:
	mode -- the PIL mode that the image should be converted to
		(RGB for color or L for grayscale)
	"""
	image = PIL.Image.open(path)
	image = image.convert(mode)
	image = np.array(image)
	# squash
	image = scipy.misc.imresize(image, (height, width), 'bilinear')
	if image.ndim == 2:
		image = image[:,:, np.newaxis]
		
	return transformer.preprocess('data', image)

class CaffeWorker(threading.Thread):

	def __init__(self, group=None, target=None, name=None,
				 args=(), kwargs=None, verbose=None):
		threading.Thread.__init__(self, group=group, target=target, name=name,
								  verbose=verbose)
		self.args = args
		self.kwargs = kwargs
		self.outputs = None
		self.net , self.transformer, self.q = args
		return

	def run(self):
		dims = self.transformer.inputs['data'][1:]
		processed = 0
		while True:
			chunk = self.q.get()
			new_shape = (len(chunk),) + tuple(dims)
			if self.net.blobs['data'].data.shape != new_shape:
				self.net.blobs['data'].reshape(*new_shape)
			for index, image in enumerate(chunk):
				self.net.blobs['data'].data[index] = image
			output = self.net.forward()[self.net.outputs[-1]]
			
			if self.outputs is None:
				self.outputs = np.copy(output)
			else:
				self.outputs = np.vstack((self.outputs, output))
			processed += len(chunk)
			print 'Processed %d images ...' % processed
			self.q.task_done()
	

def forward_all(deploy_file, caffemodel, image_list, root_folder=None, nogpu=False, mean_file=None, mean_pixel=None, **kwargs):
	q = Queue.Queue()

	if not nogpu:
		caffe.set_mode_gpu()
	
	net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
	transformer = get_transformer(deploy_file, mean_file, mean_pixel)
	
	# setting mean_pixel
	# transformer.set_mean('data', np.array([97.314453125, 114.51234436, 117.263778687]))
	
	t = CaffeWorker(args=(net, transformer, q))
	t.daemon = True
	t.start()
	
	batch_size, channels, height, width = transformer.inputs['data']
	
	# remove label if present
	image_urls = []
	for line in open(image_list):
		line = line.rstrip('\n')
		chunks = line.split()
		if len(chunks) > 1:
			line = " ".join(chunks[0:-1])
		if root_folder is not None:
			line = root_folder.rstrip("/") + "/" + line
		image_urls.append(line)
	
	for urls_chunk in [image_urls[x:x+batch_size] for x in xrange(0, len(image_urls), batch_size)]:
		chunk = np.array([load_and_prepare_image(transformer, url, height, width) for url in urls_chunk])
		q.put(chunk)
		
	print 'Preprocessing done ...'
	q.join()
	return t.outputs

if __name__ == '__main__':
	os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
	script_start_time = time.time()
	parser = argparse.ArgumentParser(description='Queued features extraction with GPUs')

	### Positional arguments

	parser.add_argument('deploy_file',  help='Path to the deploy file')
	parser.add_argument('caffemodel',   help='Path to a .caffemodel')
	parser.add_argument('image_list',   help='Path to an image list')
	parser.add_argument('output_file',  help='Name of output file')

	### Optional arguments

	parser.add_argument('-mf', '--mean-file',
			help='Path to a mean file (*.binaryproto)')
	parser.add_argument('-mp', '--mean-pixel',
			help='Path to a mean pixel numpy file (*.npy)')
	parser.add_argument('--nogpu',
			action='store_true',
			help="Don't use the GPU")
	parser.add_argument('-rf', '--root-folder',
			default='',
			help='Root folder of images in the list.')

	args = vars(parser.parse_args())
	
	# out = forward_all(args['deploy_file'], args['caffemodel'], args['image_list'], not args['nogpu'], args['mean_file'], args['mean_pixel'])
	out = forward_all(**args);
	
	print 'Saving to %s ...' % args['output_file']
	if args['output_file'].endswith('bin') or args['output_file'].endswith('dat'):
		out.tofile(args['output_file'])
	else:
		np.save(args['output_file'], out)
	print 'Script took %s seconds.' % (time.time() - script_start_time,)
