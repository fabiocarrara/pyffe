import os

import pyffe


# DataSubset is DEPRECATED
class DataSubset(object):
    def __init__(self, parent, list_file_name):
        self.parent = parent
        self.list_file = list_file_name
        self.list_name = os.path.splitext(self.list_file)[0]
        self.list_absolute_path = self.parent.path + "/" + self.list_file
        self.count = None

        # FIXME Vars for deserialization
        # Used for transition from DataSubset to ListFile
        self.urls = None
        self.labels = None
        self.abs_path = None
        self._loaded = False

        self.get_count()

    def __getattr__(self, name):
        if hasattr(self.__dict__, name):
            return self.__dict__[name]
        elif 'parent' in self.__dict__ and hasattr(self.__dict__['parent'], name):
            return self.__dict__['parent'].__dict__[name]

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


class Dataset(object):
    def __init__(self, dataset_path):
        self.path = os.path.abspath(dataset_path).rstrip('/')
        self.name = os.path.basename(self.path)
        self.root_folder = None

        config_file = self.path + "/config.py"
        if os.path.exists(config_file):
            context = dict()
            execfile(config_file, context)
            self.__dict__.update(context['config'])
            
        self.load_subsets()

    def load_subsets(self):
        self.subsets = {
            os.path.splitext(list_file)[0]: pyffe.ListFile(self.path + '/' + list_file, self)
            for list_file in os.listdir(self.path) if list_file.endswith(".txt")
            }

    def __getattr__(self, name):
        if hasattr(self.__dict__, name):
            return self.__dict__[name]
        elif 'subsets' in self.__dict__ and name in self.__dict__['subsets']:
            return self.__dict__['subsets'][name]

        raise AttributeError("No attribute called {} is present".format(name))
