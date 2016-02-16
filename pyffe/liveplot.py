#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class LivePlot():

	def __init__(self, **kwargs):
		plt.ion()
		plt.rc('lines', marker='.')
		
		# Set up plot
		gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
		self.figure = plt.figure()
		self.axup = plt.subplot(gs[0])
		self.axup2 = self.axup.twinx()
		self.axdown = plt.subplot(gs[1])
		
		plt.tight_layout()
		if 'title' in kwargs: plt.title(kwargs['title'])
		
		# Will contain lines
		self.lines = dict()

		# Autoscale on unknown axis and known lims on the other
		self.axup.set_autoscalex_on(True)
		self.axup.set_autoscaley_on(True)
		self.axup.set_prop_cycle(cycler('color', ['g', 'b', 'r', 'c']))
		
		self.axdown.set_autoscalex_on(True)
		self.axdown.set_autoscaley_on(True)
		
		self.axup2.set_autoscalex_on(True)
		self.axup2.set_autoscaley_on(False)
		self.axup2.set_ylim(ymin=0, ymax=1)
		self.axup2.set_yticks(np.arange(0,11) / 10.0)
		self.axup2.set_yticks(np.arange(0,21) / 20.0, minor=True)
		self.axup2.set_prop_cycle(cycler('color', ['darkorange', 'm', 'k', 'y']))
		
		# Other stuff
		self.axup2.grid(which='major')
		self.axup2.grid(which='minor', alpha=0.2)

	def update_data(self, prefix, label, axis, xdata, ydata):
		name = prefix + '_' + label
		# Get existing line or create it
		if name not in self.lines:
			self.lines[name], = axis.plot([],[], label=name)

		# Update data (with the new _and_ the old points)
		self.lines[name].set_xdata(xdata)
		self.lines[name].set_ydata(ydata)
		# Need both of these in order to rescale
		axis.relim()
		axis.autoscale_view()
		# We need to draw *and* flush
		self.figure.canvas.draw()
		self.figure.canvas.flush_events()

	def waitclose(self):
		plt.ioff()
		plt.show()

	def __call__(self, data):
		# TRAIN DATA PLOTTING
		times = data['train']['time']
		its = data['train']['iteration']
		num_its = len( its )
		if num_its > 0:
			# AVG LOSS
			num_dat = len( data['train']['avg_loss'] )
			self.update_data('train', 'avg_loss', self.axup, its[:num_dat], data['train']['avg_loss'])
			
			# TRAIN OUTPUTS
			for label, dat in data['train']['out'].iteritems():
				# FIXME ugly suppression of train loss
				if label == 'loss': continue
				num_dat = len( dat )
				ax = self.axup if label == 'loss' else self.axup2
				self.update_data('train', label, ax, its[:num_dat], dat)
				
			# LEARNING RATE
			num_dat = len( data['train']['lr'] )
			self.update_data('train', 'lr', self.axdown, its[:num_dat], data['train']['lr'])
			
		
		# TEST DATA PLOTTING
		times = data['test']['time']
		its = data['test']['iteration']
		num_its = len( its )
		if num_its > 0:
			for test_num, outs in data['test']['out'].iteritems():
				for label, dat in outs.iteritems():
					num_dat = len( dat )
					ax = self.axup if label == 'loss' else self.axup2
					self.update_data('test', label+"#"+str(test_num), ax, its[:num_dat], dat)
		
		# self.ax1.legend(self.lines, [l.get_label() for k,l in self.lines.iteritems()], loc=0)
		self.axup.legend(loc='center right', fancybox=True)
		self.axup2.legend(loc='upper left', fancybox=True)
		self.axdown.legend(loc='upper right', fancybox=True)
	
if __name__ == "__main__":
	import os
	import sys
	import pyffe
	if len(sys.argv) < 2:
		print "Usage: {} log.caffelog".format(os.path.basename(sys.argv[0]))
		sys.exit(1)
		
	with open(sys.argv[1], "r") as f:
		lp = LivePlot()
		line_iter = iter( f.readline, '' )
		data = pyffe.LogParser(line_iter).parse()
		lp(data)
		lp.waitclose()
