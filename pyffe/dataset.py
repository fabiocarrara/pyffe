import os


class DataSubset(object):

	def __init__(self, parent, list_file_name):
		self.parent = parent
		self.list_file = list_file_name
		self.list_name = os.path.splitext(self.list_file)[0]
		self.list_absolute_path = self.parent.path + "/" + self.list_file
		self.count = None
		self.get_count()

	def __getattr__(self, name):
		if hasattr(self.parent, name):
			return self.parent.__dict__[name]

		raise AttributeError("No attribute called {} is present".format(name))

	def get_count(self):
		if self.count is not None:
			return self.count

		with open(self.parent.path + '/' + self.list_file) as f:
			for i, l in enumerate(f):
				pass
		self.count = i + 1
		return self.count

	def get_list_full_path(self):
		return self.parent.path + "/" + self.list_file

	def get_name(self):
		return self.parent.name + '_' + self.list_name

	def __str__(self):
		return self.get_name()


class Dataset (object):

	def __init__(self, dataset_path):
		self.path = os.path.abspath(dataset_path)
		self.name = os.path.basename(self.path.rstrip("/"))

		config_file = self.path + "/config.py"
		if os.path.exists(config_file):
			execfile(config_file, self.__dict__)

		self.subsets = {
			os.path.splitext(list_file)[0]: DataSubset(self, list_file)
			for list_file in os.listdir(self.path) if list_file.endswith(".txt")
		}

	def __getattr__(self, name):
		if hasattr(self, name):
			return self.__dict__[name]
		elif name in self.subsets:
			return self.subsets[name]

		raise AttributeError("No attribute called {} is present".format(name))