import os
import re
import sys
import glob
import pyffe
import pickle
import signal
import logging
import numpy as np
import subprocess
from subprocess import call

logging.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# utilitiy functions
def mkdir_p(path):
	if not os.path.exists(path):
		os.makedirs(path)

class Experiment (object):

	SNAPSHOTS_DIR	= "snapshots"
	LOG_FILE 		= "log.caffelog"

	def __init__(self, pyffe_model, pyffe_solver, train, test=[], val=[]):
		self.model = pyffe_model
		self.solver = pyffe_solver
		
		self.train = train
		self.test = test if type(test) is list else [test]
		self.val = val if type(val) is list else [val]
		
	def short_name(self):
		name = self.model.name + "-tr_" + self.train.get_name()
		if self.val is not None:
			name = name + "-vl_" + "_".join([v.get_name() for v in self.val])
		if self.test is not None:
			name = name + "-ts_" + "_".join([t.get_name() for t in self.test])
		return name
	
	def long_name(self):
		name = self.model.name + " trained on " + self.train.get_name()
		if self.val is not None:
			name = name + ", validated on " + ", ".join([v.get_name() for v in self.val])
		if self.test is not None:
			name = name + ", tested on " + "_".join([t.get_name() for t in self.test])
		return name
		
	def setup(self, exps_parent_dir):
		logging.info("Setting up " + self.long_name() + " ...")
		self.name = self.short_name()
		self.workdir = exps_parent_dir.rstrip('/') + '/' + self.name
		
		mkdir_p(self.workdir + "/" + self.SNAPSHOTS_DIR)
				
		if os.path.exists(self.workdir + "/" + self.LOG_FILE):
			os.remove(self.workdir + "/" + self.LOG_FILE)
			
		self.workdir = os.path.abspath(self.workdir)
		
		## SETUP TRAIN
		
		with open(self.workdir + "/train.prototxt", "w") as f:
			f.write(self.model.to_train_prototxt(self.train))
		
		self.solver.set_train("train.prototxt", self.train.get_count(), self.model.get_train_batch_size())
		
		## VAL
		for v in self.val:
			val_file = "val-" + v.get_name() + ".prototxt"
			with open(self.workdir + "/" + val_file, "w") as f:
				f.write(self.model.to_val_prototxt(v))

			self.solver.add_val(val_file, v.get_count(), self.model.get_val_batch_size())
			
		with open(self.workdir + "/deploy.prototxt", "w") as f:
			f.write(self.model.to_deploy_prototxt())
		
		with open(self.workdir + "/solver.prototxt", "w") as f:
			f.write(self.solver.to_solver_prototxt())
			
		# WRITE OR LINK MEAN IMAGE / MEAN PIXEL / INITIAL WEIGHTS
		
		if self.model.infmt.mean_pixel is not None:
			np.save( self.workdir + "/mean-pixel.npy", np.array(self.model.infmt.mean_pixel) )
			
		if self.model.pretrain is not None:
			os.system("ln -s -r " + self.model.pretrain + " " + self.workdir)
		
		# DRAW NET
		
		os.system("/opt/caffe/python/draw_net.py --rankdir TB " + self.workdir + "/train.prototxt " + self.workdir + "/net.png > /dev/null")
		
		# DUMP EXPERIMENT OBJ
		# FIXME does not work...
		#with open(self.workdir + "/exp.pyffe", "w") as f:
		#	pickle.dump(self, f)
	
	def run(self):
		logging.info("Training on " + self.train.get_name() + " while validating on " + ", ".join([ str(v) for v in self.val ]) + " ...")
		os.chdir(self.workdir)
		
		cmd = ["caffe", "train", "-gpu", "0", "-solver", "solver.prototxt"]		
		if self.model.infmt.pretrain is not None:
			cmd = cmd + ["-weights", os.path.basename(self.model.infmt.pretrain)]
			
		caffe = subprocess.Popen(cmd, stderr=subprocess.PIPE)
		tee = subprocess.Popen(["tee", "-a", self.LOG_FILE], stdin=caffe.stderr, stdout=subprocess.PIPE)
		
		def handler(signal, frame):
			# propagate SIGINT down, and wait
			os.kill(caffe.pid, signal)
			caffe.wait()
		
		signal.signal(signal.SIGINT, handler)
		
		line_iter = iter( tee.stdout.readline, '' )
		liveplot = pyffe.LivePlot(title=self.long_name())
		pyffe.LogParser(line_iter).parse(liveplot)
		
	def run_test(self):
		os.chdir(self.workdir)
		if not self.test: # no tests
			logging.info("No test defined for this experiment {}".format(self.long_name()))
			return
		
		# find last snapshot
		p = re.compile("\d+")
		maxiter = str( max([ int(p.findall(sn)[0]) for sn in glob.glob("snapshots/*.caffemodel") ]) )
	
		for t in self.test:
			logging.info("Testing on " + t.get_name() + " ...")
			test_file = "test-" + t.get_name() + ".prototxt"
			
			with open(self.workdir + "/" + test_file, "w") as f:
				net, iters = self.model.to_test_prototxt(t)
				f.write(net)
			
			# TODO python data layer with async blob preparation
			os.system("caffe test -gpu 0 -model {} -weights snapshots/snapshot_iter_{}.caffemodel -iterations {} 2> test-{}.caffelog".format(test_file, maxiter, iters, t.get_name()))


