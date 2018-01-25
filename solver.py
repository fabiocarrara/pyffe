import copy
import math
import caffe
from google.protobuf.text_format import Merge


class Solver(object):
    def __init__(self, **kwargs):
        self.__dict__['params'] = dict(
            train_epochs=3,
            val_interval_epochs=1,
            val_epochs=1,
            snapshot_interval_epochs=1,
            display_per_epoch=50,
            lr_policy="fixed",
            stepsize_epochs=1
        )
        self.params.update(kwargs)

        self.init_solver_parameter()

        # pass args to SolverParameter
        for k, v in self.params.iteritems():
            if hasattr(self.sp, k):
                setattr(self.sp, k, v)

    def __getstate__(self):
        state = copy.copy(self.__dict__)
        # Binary serialization is more space-efficient, but introduces
        # approximation errors on floats, sticking with string representation
        # state['sp'] = self.__dict__['sp'].SerializeToString()
        state['sp'] = str(self.__dict__['sp'])
        return state

    def __setstate__(self, state):
        object.__setattr__(self, '__dict__', state)
        tmp = self.__dict__['sp']
        self.__dict__['sp'] = caffe.proto.caffe_pb2.SolverParameter()
        # self.__dict__['sp'].ParseFromString(tmp)
        Merge(tmp, self.__dict__['sp'])

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.params:
            return self.params[name]

        raise AttributeError("No attribute called {} is present".format(name))

    def __setattr__(self, name, value):
        if name in self.__dict__['params']:
            self.__dict__['params'][name] = value

        if 'sp' in self.__dict__ and hasattr(self.__dict__['sp'], name):
            setattr(self.__dict__['sp'], name, value)

    def init_solver_parameter(self):
        self.__dict__['sp'] = caffe.proto.caffe_pb2.SolverParameter()

        self.sp.solver_type = 0  # SGD
        self.sp.solver_mode = 1  # GPU

        # critical:
        self.sp.base_lr = 0.01
        self.sp.momentum = 0.9

        # speed:
        # self.sp.test_iter = 100
        # self.sp.test_interval = 250

        # looks:
        # self.sp.display = 25
        # self.sp.snapshot = 2500
        self.sp.snapshot_prefix = "snapshots/snapshot"

        # learning rate policy
        self.sp.lr_policy = "fixed"

        # important, but rare:
        self.sp.gamma = 0.1
        self.sp.weight_decay = 0.0005
        # self.sp.train_net = trainnet_prototxt_path
        # self.sp.test_net = testnet_prototxt_path

        # pretty much never change these.
        # self.sp.max_iter = 100000
        self.sp.test_initialization = False  # validation disabled by default
        # self.sp.average_loss = 25 # this has to do with the display.
        # self.sp.iter_size = 1 # this is for accumulating gradients
        self.sp.random_seed = 23

    def set_train(self, proto_path, num, batch_size):
        self.sp.train_net = proto_path
        self.sp.max_iter = int(math.ceil(float(self.params['train_epochs']) * num / batch_size))
        self.sp.display = int(max(round(num / (batch_size * float(self.params['display_per_epoch']))), 1))
        self.sp.average_loss = self.sp.display  # display averaged loss
        self.sp.snapshot = int(math.ceil(float(self.params['snapshot_interval_epochs']) * num / batch_size))
        self.sp.test_interval = int(math.ceil(float(self.params['val_interval_epochs']) * num / batch_size))

        if self.sp.lr_policy == 'step':
            self.sp.stepsize = int(math.ceil(float(self.params['stepsize_epochs']) * num / batch_size))

    def set_train_val(self, proto_path, train, t_batch_size, vals, v_batch_size):
        t_num = train.get_count()

        # TRAIN SETUP
        self.sp.net = proto_path
        self.sp.max_iter = int(math.ceil(float(self.params['train_epochs']) * t_num / t_batch_size))
        self.sp.display = int(max(round(t_num / (t_batch_size * float(self.params['display_per_epoch']))), 1))
        self.sp.average_loss = self.sp.display  # display averaged loss
        self.sp.snapshot = int(math.ceil(float(self.params['snapshot_interval_epochs']) * t_num / t_batch_size))
        self.sp.test_interval = int(math.ceil(float(self.params['val_interval_epochs']) * t_num / t_batch_size))

        if self.sp.lr_policy == 'step':
            self.sp.stepsize = int(math.ceil(float(self.params['stepsize_epochs']) * t_num / t_batch_size))

        # TEST SETUP
        self.sp.test_initialization = True
        for v in vals:
            self.sp.test_iter.append(int(math.ceil(float(self.params['val_epochs']) * v.get_count() / v_batch_size)))
            netState = self.sp.test_state.add()
            netState.stage.append('val-on-' + v.get_name())

    def add_val(self, proto_path, num, batch_size):
        self.sp.test_initialization = True
        self.sp.test_net.append(proto_path)
        self.sp.test_iter.append(int(math.ceil(float(self.params['val_epochs']) * num / batch_size)))

    def disable_val(self):
        del self.sp.test_net[:]
        del self.sp.test_iter[:]
        self.sp.test_initialization = False

    def stop_and_snapshot_at(self, last_iter, snapshot_iter):
        self.disable_val()
        self.sp.snapshot = snapshot_iter
        self.sp.max_iter = last_iter
        self.sp.test_initialization = False

    def to_solver_prototxt(self):
        return str(self.sp)
