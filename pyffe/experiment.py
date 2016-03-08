import os
import re
import sys
import copy
import glob
import pyffe
import pickle
import signal
import shutil
import logging
import numpy as np
import pandas as pd
import subprocess
from subprocess import call

logging.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# utilitiy functions
def mkdir_p(path):
	if not os.path.exists(path):
		os.makedirs(path)
		
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

# exported functions
def load(path):
	# path can be the path to the .pyffe file, to the folder containing it,
	# or to the folder of experiments
	
	# pyffe file
	if os.path.isfile(path):
		return pickle.load(open(path, 'rb'))
		
	if os.path.isdir(path):
		# seach for pyffe file
		for filename in os.listdir(path):
			filename = path.rstrip('/') + '/' + filename
			if os.path.isfile(filename) and filename.endswith('.pyffe'):
				return pickle.load(open(filename, 'rb'))
				
		# if not found, maybe is an experiments collection dir
		return [ pickle.load(open(path.rstrip('/') + '/' + filename + '/' + Experiment.EXP_FILE, 'rb')) for filename in os.listdir(path) if os.path.isdir(path+'/'+filename) ]

def summarize(exps):
	report = pd.DataFrame()
	for e in exps:
		r = e.summarize()
		report = report.append(r)
	report = report.fillna('-')
	with pd.option_context('expand_frame_repr', False):
		print report
		report.to_csv('summary.csv')

class Experiment (object):

	SNAPSHOTS_DIR	= 'snapshots'
	LOG_FILE 		= 'log.caffelog'
	EXP_FILE		= 'exp.pyffe'

	def __init__(self, pyffe_model, pyffe_solver, train, test=[], val=[]):
		self.model = copy.deepcopy(pyffe_model)
		self.solver = copy.deepcopy(pyffe_solver)
		
		self.train = train
		self.test = test if type(test) is list else [test]
		self.val = val if type(val) is list else [val]
		
	def short_name(self):
		name = self.model.name + '-tr_' + self.train.get_name()
		if self.val is not None:
			name = name + '-vl_' + '_'.join([v.get_name() for v in self.val])
		if self.test is not None:
			name = name + '-ts_' + '_'.join([t.get_name() for t in self.test])
		
		if len(name) > 150: # name too long
			name = self.model.name + '-tr_' + self.train.get_name()
			if self.val is not None:
				name = name + '-vl_' + self.val[0].get_name() + '_etcEtc_' + self.val[-1].get_name()
			if self.test is not None:
				name = name + '-ts_' + self.test[0].get_name() + '_etcEtc_' + self.test[-1].get_name()
				
		return name
	
	def long_name(self):
		name = self.model.name + ' trained on ' + self.train.get_name()
		if self.val is not None:
			name = name + ', validated on ' + ', '.join([v.get_name() for v in self.val])
		if self.test is not None:
			name = name + ', tested on ' + '_'.join([t.get_name() for t in self.test])
		return name
		
	def clean(self):
		raise NotImplementedError()
		
	def setup(self, exps_parent_dir):
		logging.info('Setting up ' + self.long_name() + ' ...')
		self.name = self.short_name()
		self.workdir = exps_parent_dir.rstrip('/') + '/' + self.name
		
		mkdir_p(self.workdir + '/' + self.SNAPSHOTS_DIR)
			
		self.workdir = os.path.abspath(self.workdir)
		
		## SETUP TRAIN
		
		with open(self.workdir + '/train.prototxt', 'w') as f:
			f.write(self.model.to_train_prototxt(self.train))
		
		self.solver.set_train('train.prototxt', self.train.get_count(), self.model.get_train_batch_size())
		
		## VAL
		for v in self.val:
			val_file = 'val-' + v.get_name() + '.prototxt'
			with open(self.workdir + '/' + val_file, 'w') as f:
				f.write(self.model.to_val_prototxt(v))

			self.solver.add_val(val_file, v.get_count(), self.model.get_val_batch_size())
			
		with open(self.workdir + '/deploy.prototxt', 'w') as f:
			f.write(self.model.to_deploy_prototxt())
		
		with open(self.workdir + '/solver.prototxt', 'w') as f:
			f.write(self.solver.to_solver_prototxt())
			
		# WRITE OR LINK MEAN IMAGE / MEAN PIXEL / INITIAL WEIGHTS
		
		if self.model.infmt.mean_pixel is not None:
			np.save( self.workdir + '/mean-pixel.npy', np.array(self.model.infmt.mean_pixel) )
			
		if self.model.pretrain is not None:
			os.system('ln -s -r ' + self.model.pretrain + ' ' + self.workdir)
		
		# DRAW NET
		
		os.system('/opt/caffe/python/draw_net.py --rankdir TB ' + self.workdir + '/train.prototxt ' + self.workdir + '/net.png > /dev/null')
		
		# DUMP EXPERIMENT OBJ
		with open(self.workdir + '/' + self.EXP_FILE, 'w') as f:
			pickle.dump(self, f)
	
	def run(self, live_plot=True):
		logging.info('Training on ' + self.train.get_name() + ' while validating on ' + ', '.join([ str(v) for v in self.val ]) + ' ...')
		os.chdir(self.workdir)
		
		if os.path.exists(self.LOG_FILE):
			os.remove(self.LOG_FILE)
		
		cmd = ['caffe', 'train', '-gpu', '0', '-solver', 'solver.prototxt']		
		if self.model.infmt.pretrain is not None:
			cmd = cmd + ['-weights', os.path.basename(self.model.infmt.pretrain)]
				
		caffe = subprocess.Popen(cmd, stderr=subprocess.PIPE)
		
		dst = subprocess.PIPE if live_plot else open(os.devnull, 'wb')
		
		tee = subprocess.Popen(['tee', '-a', self.LOG_FILE], stdin=caffe.stderr, stdout=dst)
		
		def handler(signal, frame):
			# propagate SIGINT down, and wait
			os.kill(caffe.pid, signal)
			caffe.wait()
		
		signal.signal(signal.SIGINT, handler)
		
		if live_plot:
			line_iter = iter( tee.stdout.readline, '' )
			liveplot = pyffe.LivePlot(title=self.long_name())
			pyffe.LogParser(line_iter).parse(liveplot)
			
		tee.wait()
		
	def run_test(self):
		os.chdir(self.workdir)
		if not self.test: # no tests
			logging.info('No test defined for this experiment {}'.format(self.long_name()))
			return
		
		# find last snapshot
		p = re.compile('\d+')
		maxiter = str( max([ int(p.findall(sn)[0]) for sn in glob.glob('snapshots/*.caffemodel') ]) )
	
		for t in self.test:
			logging.info('Testing on ' + t.get_name() + ' ...')
			test_file = 'test-' + t.get_name() + '.prototxt'
			
			with open(self.workdir + '/' + test_file, 'w') as f:
				net, iters = self.model.to_test_prototxt(t)
				f.write(net)
			
			# TODO python data layer with async blob preparation
			os.system('caffe test -gpu 0 -model {} -weights snapshots/snapshot_iter_{}.caffemodel -iterations {} 2> test-{}.caffelog'.format(test_file, maxiter, iters, t.get_name()))
	
	def get_log_data(self):
		line_iter = iter( open(self.workdir + '/' + self.LOG_FILE).readline, '' )
		return pyffe.LogParser(line_iter).parse()
	
	def show_logs(self):
		plot = pyffe.LivePlot(
			title=self.long_name(),
			train=self.train,
			val=self.val
		)
		plot(self.get_log_data())
	
	def print_test_results(self):
		print
		print self.long_name()
		print '==============='
		
		for t in self.test:
			print t.get_name(), ':'
			test_file = self.workdir + '/test-' + t.get_name() + '.caffelog'
			os.system('grep "accuracy =" {} | grep -v Batch'.format(test_file))
	
	def summarize(self, show_train_points=True):
		log_data = self.get_log_data()
		last_iter = log_data['train']['iteration'][-1]
		bs = log_data['meta']['batch_size'][0]
		
		# list of indices where max accuracies for each test are
		itmax = [ argmax(outs['accuracy']) for k, outs in log_data['test']['out'].iteritems() ]
		
		pdata = [ [round(outs['accuracy'][i],2) for i in itmax] for k, outs in log_data['test']['out'].iteritems() ]
		vnames = [v.get_name() for v in self.val]
		
		v_idx_num = len(self.val)
		v_idx_names = vnames
		
		if show_train_points:
			train_pcent = [ '{0:.0f}% (~{1} imgs)'.format(100 * log_data['test']['iteration'][i] / last_iter, log_data['test']['iteration'][i]*bs ) for i in itmax ]
			pdata = pdata + [ train_pcent ]
			v_idx_num = len(self.val) + 1
			v_idx_names = vnames + ['   --> at']
			
		index = [
			[self.model.name] * v_idx_num,
			[self.train.get_name()] * v_idx_num,
			v_idx_names
		]
		
		return pd.DataFrame(pdata, index=index, columns=vnames)
		
	def extract_features(self, dataset, snapshot_iter=None, blobname=None):
		os.chdir(self.workdir)
		net, last_top, iters = self.model.to_extract_prototxt(dataset)
		
		if blobname is None:
			blobname = last_top
			
		if snapshot_iter is None:
			p = re.compile('\d+')
			snapshot_iter = str( max([ int(p.findall(sn)[0]) for sn in glob.glob('snapshots/*.caffemodel') ]) )
			
		logging.debug('Extracting \'{}\' features from {} using snapshots/snapshot_iter_{}.caffemodel'.format(blobname, dataset.get_name(), snapshot_iter))
		
		extract_file = 'extract-' + dataset.get_name() + '.prototxt'
		with open(extract_file, 'w') as f:
				f.write(net)
		
		lmdb_name = dataset.get_name() + '_' + blobname
		if os.path.exists(lmdb_name):
			shutil.rmtree(lmdb_name)
		
		os.system('caffe-extract-features snapshots/snapshot_iter_{}.caffemodel {} {} {} {} lmdb GPU 0'.format(snapshot_iter, extract_file, blobname, lmdb_name, iters))
		
	def get_features(self, lmdb_name):
		os.chdir(self.workdir)
		import lmdb
		import caffe
		feats = None
		datum = caffe.proto.caffe_pb2.Datum()
		with lmdb.open(lmdb_name) as env:
			with env.begin() as txn:
				with txn.cursor() as c:
					for k,v in c:
						datum.ParseFromString(v)
						#print datum.channels, datum.height, datum.width, datum.label
						feat = caffe.io.datum_to_array(datum).squeeze()
						#print feat.shape
						if feats is not None:
							feats = np.vstack((feats, feat))
						else:
							feats = feat
		return feats
