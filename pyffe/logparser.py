#!/usr/bin/env python
#-*- coding:utf-8 -*-
import re

class LogParser (object):

	batch_size = re.compile('.*output data size: (\d+),.*')
	
	snapshots = re.compile('.*_iter_(\d+).caffemodel.*')

	test_it = re.compile('I(\d{2})(\d{2}) (\d+):(\d+):(\d+)\.(\d{3}).*Iteration (\d+), Testing net \(#(\d+)\)')
	test_out = re.compile('.*Test net output #(\d+): (\w+) = ([^\s]*)( \(.*)?')
	
	train_it = re.compile('I(\d{2})(\d{2}) (\d+):(\d+):(\d+)\.(\d{3}).*Iteration (\d+), loss = (.*)')
	train_out = re.compile('.*Train net output #(\d+): (\w+) = ([^\s]*)( \(.*)?')
	train_lr = re.compile('.*Iteration \d+, lr = (.*)')

	def __init__(self, line_iterator):
		self.line_iter = line_iterator
		self.data = dict(
			meta = dict(batch_size=[], snapshots=[]),
			train = dict(time=[], iteration=[], lr=[], avg_loss=[], out=dict()),
			test  = dict(time=[], iteration=[], out=dict())
		)
		
		self.t_zero = None
		
	def get_relative_time(self, month, day, hour, minute, second, millis):
		time_seconds = millis + 1000*(second + 60*(minute + 60*(hour + 24*(day + 30*month))))
		if self.t_zero is None:
			self.t_zero = time_seconds
			
		return time_seconds - self.t_zero
		
	def parse(self, callback=None):
		test_num = 0
		for line in self.line_iter:
			# BATCH SIZE
			matches = self.batch_size.match(line)
			if matches is not None:
				bs = int( matches.group(1) )
				self.data['meta']['batch_size'].append(bs)
			
			# SNAPSHOT
			matches = self.snapshots.match(line)
			if matches is not None:
				ss_iter = int( matches.group(1) )
				self.data['meta']['snapshots'].append(ss_iter)
			
			# NEW TEST ITERATION
			matches = self.test_it.match(line)
			if matches is not None:
				month = int( matches.group(1) )
				day = int( matches.group(2) )
				hour = int( matches.group(3) )
				minute = int( matches.group(4) )
				second = int( matches.group(5) )
				millis = int( matches.group(6) )
				it = int( matches.group(7) )
				test_num = int( matches.group(8) )
				
				time = self.get_relative_time(month, day, hour, minute, second, millis)
				
				if test_num is 0:
					self.data['test']['time'].append(time)
					self.data['test']['iteration'].append(it)
			
			# TEST OUTPUTS
			matches = self.test_out.match(line)
			if matches is not None:
				out_num = int( matches.group(1) ) # maybe useless?
				out_name = str( matches.group(2) )
				out_value = float( matches.group(3) )
				
				if test_num not in self.data['test']['out']:
					self.data['test']['out'][test_num] = dict()
				
				if out_name not in self.data['test']['out'][test_num]:
					self.data['test']['out'][test_num][out_name] = []
					
				self.data['test']['out'][test_num][out_name].append(out_value)
				
			# NEW TRAIN ITERATION
			matches = self.train_it.match(line)
			if matches is not None:
				month = int( matches.group(1) )
				day = int( matches.group(2) )
				hour = int( matches.group(3) )
				minute = int( matches.group(4) )
				second = int( matches.group(5) )
				millis = int( matches.group(6) )
				
				time = self.get_relative_time(month, day, hour, minute, second, millis)
				
				it = int( matches.group(7) )
				avg_loss = float( matches.group(8) )
				
				self.data['train']['time'].append(time)
				self.data['train']['iteration'].append(it)
				self.data['train']['avg_loss'].append(avg_loss)
			
			# TRAIN OUTPUTS
			matches = self.train_out.match(line)
			if matches is not None:
				out_num = int( matches.group(1) ) # maybe useless?
				out_name = str( matches.group(2) )
				out_value = float( matches.group(3) )
				
				if out_name not in self.data['train']['out']:
					self.data['train']['out'][out_name] = []
					
				self.data['train']['out'][out_name].append(out_value)
			
			# LEARNING RATE
			matches = self.train_lr.match(line)
			if matches is not None:
				lr = float( matches.group(1) )
				self.data['train']['lr'].append(lr)
				
			if callback is not None:
				callback(self.data)
		
		return self.data

if __name__ == "__main__":
	import pyffe
	import matplotlib.pyplot as plt
	
	with open("train.caffelog", "r") as f:
		parser = LogParser(iter(f.readline, ''))
		lp = pyffe.LivePlot()
		parser.parse(lp)
		lp.waitclose()
	
