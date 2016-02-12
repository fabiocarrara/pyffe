import os
import re
import glob
import pyffe
import pickle
import numpy as np
import subprocess
from subprocess import call

class Experiment (object):

	# TODO val as list using stages, test as list after training
	def __init__(self, pyffe_model, pyffe_solver, train, test=None, val=None):
		self.model = pyffe_model
		self.solver = pyffe_solver
		
		self.train = train
		self.test = test
		self.val = val if val is not None else train
		
	def short_name(self):
		name = self.model.params['name'] + "-tr_" + self.train.get_name()
		if self.val is not None:
			name = name + "-vl_" + self.val.get_name()
		if self.test is not None:
			name = name + "-ts_" + self.test.get_name()
		return name
	
	def long_name(self):
		name = self.model.params['name'] + " trained on " + self.train.get_name()
		if self.val is not None:
			name = name + ", validated on" + self.val.get_name()
		if self.test is not None:
			name = name + ", tested on" + self.test.get_name()
		return name
		
	def setup(self, exps_parent_dir):
		self.name = self.short_name()
		self.workdir = exps_parent_dir.rstrip('/') + '/' + self.name
		
		for folder in ["lists", "snapshots"]:
			if not os.path.exists(self.workdir + "/" + folder):
				os.makedirs(self.workdir + "/" + folder)
				
		if os.path.exists(self.workdir + "/train.caffelog"):
			os.remove(self.workdir + "/train.caffelog")
			
		self.workdir = os.path.abspath(self.workdir)
		
		# DATASET CONSTRAINT ON MODEL AND SOLVER
		
		## TRAIN
		train_list_name = self.train.get_name() + "-" + self.train.list_file
		os.system("ln -s -r " + self.train.get_list_full_path() + " " + self.workdir + "/lists/" + train_list_name)
		
		self.model.params['train_list'] = "lists/" + train_list_name
		self.solver.set_train_epoch(self.train.get_count(), self.model.params['batch_size'])
		
		if hasattr(self.train, 'root_folder'):
			self.model.params['train_root_folder'] = self.train.root_folder
		
		## VAL
		if self.val is not None:
			val_list_name = self.val.get_name() + "-" + self.val.list_file
			os.system("ln -s -r " + self.val.get_list_full_path() + " " + self.workdir + "/lists/" + val_list_name)

			self.model.params['val_list'] = "lists/" + val_list_name
			self.solver.set_val_epoch(self.val.get_count(), self.model.params['batch_size'])

			if hasattr(self.val, 'root_folder'):
				self.model.params['val_root_folder'] = self.val.root_folder
		else:
			self.solver.disable_val();
		
		# WRITE PROTOTXT FILES		
		with open(self.workdir + "/train_val.prototxt", "w") as f:
			f.write(self.model.to_train_val_prototxt())
	
		with open(self.workdir + "/deploy.prototxt", "w") as f:
			f.write(self.model.to_deploy_prototxt())
		
		with open(self.workdir + "/solver.prototxt", "w") as f:
			f.write(self.solver.to_solver_prototxt())
			
		# SETUP TEST
		if self.test is not None:
			test_list_name = self.test.get_name() + "-" + self.test.list_file
			os.system("ln -s -r " + self.test.get_list_full_path() + " " + self.workdir + "/lists/" + test_list_name)
			self.model.params['test_list'] = "lists/" + test_list_name
			
			if hasattr(self.test, 'root_folder'):
				self.model.params['test_root_folder'] = self.test.root_folder
				
			self.model.set_max_batch_size()
			
			with open(self.workdir + "/test.prototxt", "w") as f:
				f.write(self.model.to_test_prototxt())
		
		# WRITE OR LINK MEAN IMAGE / MEAN PIXEL / INITIAL WEIGHTS
		
		if self.model.infmt.params['mean_file'] is not None:
			os.system("ln -s -r " + self.model.infmt.params['mean_file'] + " " + self.workdir)
		else:
			np.save( self.workdir + "/mean-pixel.npy", np.array(self.model.infmt.params['mean_pixel']) )
			
		if self.model.params['pretrain'] is not None:
			os.system("ln -s -r " + self.model.params['pretrain'] + " " + self.workdir)
		
		# DRAW NET
		
		os.system("/opt/caffe/python/draw_net.py --rankdir TB " + self.workdir + "/train_val.prototxt " + self.workdir + "/net.png")
		
		# DUMP EXPERIMENT OBJ
		# FIXME does not work...
		# with open(self.workdir + "/exp.pyffe", "w") as f:
			# pickle.dump(self, f)
	
	def run(self):
		os.chdir(self.workdir)
		if self.model.params['pretrain'] is not None:
			cmd = ["caffe", "train", "-gpu", "0", "-solver", "solver.prototxt", "-weights", os.path.basename(self.model.params['pretrain'])]
		else:
			cmd = ["caffe", "train", "-gpu", "0", "-solver", "solver.prototxt"]
			"| tee -a train.caffelog".format()
			
		caffe = subprocess.Popen(cmd, stderr=subprocess.PIPE)
		tee = subprocess.Popen(["tee", "-a", "train.caffelog"], stdin=caffe.stderr, stdout=subprocess.PIPE)
		
		line_iter = iter( tee.stdout.readline, '' )
		liveplot = pyffe.LivePlot(title=self.long_name())
		pyffe.LogParser(line_iter).parse(liveplot)
		
		
	def run_test(self):
		os.chdir(self.workdir)
		if self.test is None:
			print "No test defined for this experiment {}".format(self.long_name())
	
		# find last snapshot
		p = re.compile("\d+")
		maxiter = str( max([ int(p.findall(sn)[0]) for sn in glob.glob("snapshots/*.caffemodel") ]) )
		
		num = self.test.get_count()
		c = self.model.params['batch_size']
		while num % c > 10:
			c = c - 1
		self.model.params['batch_size'] = c
		iters = num / c
		
		with open("test.prototxt", "w") as f:
			f.write(self.model.to_test_prototxt())
	
		os.system("caffe test -gpu 0 -model test.prototxt -weights snapshots/snapshot_iter_{}.caffemodel -iterations {} 2>&1 | tee -a test.caffelog".format(maxiter, iters))
		
		# args = dict(
		# 	root_folder = self.train.root_folder if hasattr(self.train, 'root_folder') else None,		
		# )
		
		# if self.model.infmt.params['mean_file'] is not None:
		# 	args['mean_file'] = os.path.baseline(self.model.infmt.params['mean_file'])
		# else:
		# 	args['mean_pixel'] = "mean-pixel.npy"
			
		
		# scores = pyffe.forward_all(
		#	"deploy.prototxt",
		#	"snapshots/snapshot_iter_" + maxiter + ".caffemodel",
		#	"lists/" + self.test.list_file,
		#	**args
		# )
		
		# np.save(self.test.get_name() + ".npy", scores)


