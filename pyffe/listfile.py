import os


class ListLine(object):
    def __init__(self, url, label):
        self.url = url
        self.label = label


class ListFile(object):
    def __init__(self, path, parent=None):
        self.parent = parent

        self.urls = None
        self.labels = None
        self.count = None
        self.abs_path = os.path.abspath(path)
        self.list_name = os.path.basename(os.path.splitext(path)[0])

        self._loaded = False

    def load(self):
        if not self._loaded:
            self.urls = []
            self.labels = []
            self.count = 0
            with open(self.abs_path, 'r') as f:
                for line in f:
                    url, label = line.split(' ')
                    self.urls.append(url)
                    self.labels.append(int(label.rstrip('\n')))
                    self.count += 1

            self._loaded = True

    def __getitem__(self, key):
        self.load()
        return ListLine(self.urls[key], self.labels[key])

    def __iter__(self):
        self.load()
        for i, url in enumerate(self.urls):
            yield ListLine(url, self.labels[i])

    def __getattr__(self, key):
        # needed when depickling
        if 'parent' in self.__dict__ and self.__dict__['parent'] is not None:
            if key in self.__dict__['parent'].__dict__:
                return self.parent.__dict__[key]
        raise AttributeError("No attribute called {} is present".format(key))

    def __getstate__(self):
        self._loaded = False
        self.urls = None
        self.labels = None
        self.count = None
        return self.__dict__

    def __str__(self):
        return self.get_name()

    def get_count(self):
        self.load()
        return self.count

    def get_labels(self):
        self.load()
        return self.labels

    def get_list_absolute_path(self):
        return self.abs_path

    def get_name(self):
        if self.parent is not None:
            return self.parent.name + '_' + self.list_name
        return self.list_name

    def get_urls(self):
        self.load()
        return self.urls
