import copy
import functools
import glob
import logging
import os
import pickle
import re
import shutil
import signal
import subprocess

import numpy as np
import pandas as pd
from functools32 import lru_cache
from tqdm import tqdm

import pyffe

logging.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

#####################
# utility functions
#####################

# same as mkdir -p
def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)


# find ordinal index of max in an iterable
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


# decorator to preserve current working directory
def preserve_cwd(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        cwd = os.getcwd()
        try:
            return function(*args, **kwargs)
        finally:
            os.chdir(cwd)

    return decorator


######################
# exported functions
######################
def load(path):
    """
    Loads an experiment, or a folder of experiments.
    @param path: can be either the path to the .pyffe file, to the folder containing it,
        or to the folder of experiments
    @return: a pyffe.Experiment object or a list of them
    """

    # pyffe file
    if os.path.isfile(path):
        return pickle.load(open(path, 'rb'))

    if os.path.isdir(path):
        # search for pyffe file
        for filename in os.listdir(path):
            filename = path.rstrip('/') + '/' + filename
            if os.path.isfile(filename) and filename.endswith('.pyffe'):
                return pickle.load(open(filename, 'rb'))

        # if not found, maybe is an experiments collection dir
        return [pickle.load(open(path.rstrip('/') + '/' + filename + '/' + Experiment.EXP_FILE, 'rb')) for filename in
                os.listdir(path) if os.path.isdir(path + '/' + filename)]


def summarize(experiments):
    """
    Creates a summary of a list of experiments and dumps it to a csv file
    @param experiments: a list of pyffe.Experiment objects to be summarized
    @return: the summary as a pandas.DataFrame object
    """
    report = pd.DataFrame()
    for e in experiments:
        r = e.summarize()
        report = report.append(r)
    report = report.fillna('-')
    # with pd.option_context('expand_frame_repr', False):
    #    print report
    # report.to_csv('summary_exact_last.csv')
    return report


class Experiment(object):
    """
    Encapsulates all the data and methods needed for training, validating and testing a caffe model
    on multiple datasets.
    """
    SNAPSHOTS_DIR = 'snapshots'
    LOG_FILE = 'log.caffelog'
    EXP_FILE = 'exp.pyffe'

    # FIXME change default lists to immutable tuples ()
    def __init__(self, pyffe_model, pyffe_solver, train, test=(), val=(), tag=None):
        """
        Initialize an Experiment.
        @param pyffe_model: an object of a subclass of @see pyffe.Model describing your model
        @param pyffe_solver: a @see pyffe.Solver object
        @param train: a @see pyffe.ListFile object specifying the training samples. Usually you can get it from a @see pyffe.Dataset object
        @param test: a list of @see pyffe.ListFile objects specifying the one or more testing sets.
        @param val: a list of @see pyffe.ListFile objects specifying the one or more validation sets.
        @param tag: an optional tag to prepend to the experiment generated name
        """
        self.model = copy.deepcopy(pyffe_model)
        self.solver = copy.deepcopy(pyffe_solver)
        self.workdir = None

        self.train = train
        self.test = test if isinstance(test, (list, tuple)) else (test,)
        self.val = val if isinstance(val, (list, tuple)) else (val,)

        self.tag = tag
        self.name = None

    def short_name(self):
        """
        @return: a short description of the experiment
        """
        if self.name is not None:
            return self.name

        name = self.model.name + '-tr_' + self.train.get_name()
        if self.val is not None:
            name = name + '-vl_' + '_'.join([v.get_name() for v in self.val])
        if self.test is not None:
            name = name + '-ts_' + '_'.join([t.get_name() for t in self.test])

        if len(name) > 150:  # name too long
            name = self.model.name + '-tr_' + self.train.get_name()
            if self.val is not None:
                name = name + '-vl_' + self.val[0].get_name() + '_etcEtc_' + self.val[-1].get_name()
            if self.test is not None:
                name = name + '-ts_' + self.test[0].get_name() + '_etcEtc_' + self.test[-1].get_name()

        # add a prefix tag
        if self.tag is not None:
            name = self.tag + '-' + name

        self.name = name
        return name

    def long_name(self):
        """
        @return: a long description of the experiment
        """
        name = self.model.name + ' trained on ' + self.train.get_name()
        if self.val is not None:
            name = name + ', validated on ' + ', '.join([v.get_name() for v in self.val])
        if self.test is not None:
            name = name + ', tested on ' + '_'.join([t.get_name() for t in self.test])

        # add a prefix tag
        if self.tag is not None:
            return self.tag + '-' + name

        return name

    def get_accuracy_at(self, it, v):
        lmdb_name = self.extract_features(v, it)
        scores = self.get_features(lmdb_name)
        labels = v.get_labels()
        count = v.get_count()
        pred = np.round(scores)[:, 1]
        acc = np.sum(labels == pred) / float(count)
        return acc

    # TODO: refactor in get_all_val_argmax or something similar..
    def get_argmax_iters(self):
        log_data = self.get_log_data()
        it_idx = [argmax(outs['accuracy']) for k, outs in log_data['test']['out'].iteritems()]
        it_max = [log_data['test']['iteration'][i] for i in it_idx]
        return it_idx, it_max

    '''
    PROBABLY USELESS..
    def get_argmax_iter(self, dataset):
        log_data = self.get_log_data()
        it_idx = [(k, argmax(outs['accuracy'])) for k, outs in log_data['test']['out'].iteritems()]
        max_it_idx = max(it_idx, key=lambda x: log_data['test']['out'][x[0]][x[1]])[1]
        it_max = log_data['test']['iteration'][max_it_idx]
        return max_it_idx, it_max
    '''

    def get_max_min_val_iter(self):
        log_data = self.get_log_data()
        accuracies = np.array([outs['accuracy'] for k, outs in log_data['test']['out'].iteritems()])
        index = np.argmax(np.min(accuracies, axis=0))
        return index, log_data['test']['iteration'][index]

    def get_max_avg_val_iter(self):
        log_data = self.get_log_data()
        accuracies = np.array([outs['accuracy'] for k, outs in log_data['test']['out'].iteritems()])
        index = np.argmax(np.mean(accuracies, axis=0))
        return index, log_data['test']['iteration'][index]

    def get_last_snapshot(self):
        p = re.compile('\d+')
        snapshots = [int(p.findall(sn)[0]) for sn in
                     glob.glob(self.workdir + '/' + self.SNAPSHOTS_DIR + '/*.caffemodel')]
        return max(snapshots) if len(snapshots) is not 0 else None

    @lru_cache(maxsize=1)
    def get_log_data(self):
        line_iter = iter(open(self.workdir + '/' + self.LOG_FILE).readline, '')
        return pyffe.LogParser(line_iter).parse()

    @preserve_cwd
    def extract_features(self, dataset, snapshot_iter=None, blobname=None, force_extract=False):
        os.chdir(self.workdir)
        net, last_top, iters = self.model.to_extract_prototxt(dataset)

        if blobname is None:
            blobname = last_top

        if snapshot_iter is None:
            p = re.compile('\d+')
            snapshot_iter = str(max([int(p.findall(sn)[0]) for sn in glob.glob('snapshots/*.caffemodel')]))

        lmdb_name = dataset.get_name() + '_' + blobname + '@iter' + str(snapshot_iter)

        if not force_extract and os.path.exists(lmdb_name):
            return lmdb_name

        logging.debug('Extracting \'{}\' features from {} using snapshots/snapshot_iter_{}.caffemodel'
                      .format(blobname, dataset.get_name(), snapshot_iter))

        extract_file = 'extract-' + dataset.get_name() + '.prototxt'
        with open(extract_file, 'w') as f:
            f.write(str(net))

        if os.path.exists(lmdb_name):
            shutil.rmtree(lmdb_name)

        os.system(
            '/opt/caffe/build/tools/extract_features snapshots/snapshot_iter_{}.caffemodel {} {} {} {} lmdb GPU 0'.format(snapshot_iter,
                                                                                                                          extract_file,
                                                                                                                          blobname,
                                                                                                                          lmdb_name,
                                                                                                                          iters))

        return lmdb_name

    @preserve_cwd
    def get_features(self, lmdb_name):
        os.chdir(self.workdir)
        import lmdb
        import caffe
        feats = None
        datum = caffe.proto.caffe_pb2.Datum()
        i = 0
        with lmdb.open(lmdb_name) as env:
            count = env.stat()['entries']
            with env.begin() as txn:
                with txn.cursor() as c:
                    for k, v in tqdm(c):
                        datum.ParseFromString(v)
                        feat = caffe.io.datum_to_array(datum).squeeze()
                        if feats is None:
                            feats = np.zeros((count, feat.shape[0]), dtype=np.float32)

                        feats[i, :] = feat
                        i += 1
        return feats

    @preserve_cwd
    def forward(self, dataset, snapshot_iter=None, blobname=None):
        lmdb_name = self.extract_features(dataset, snapshot_iter, blobname)
        return self.get_features(lmdb_name)

    '''
    TO BE DEPRECATED in favour of setup_efficiently(), maintained for backward compatibility
    '''
    def setup(self, experiments_dir):
        logging.info('Setting up ' + self.long_name() + ' ...')
        self.workdir = os.path.abspath(experiments_dir.rstrip('/') + '/' + self.short_name())

        mkdir_p(self.workdir + '/' + self.SNAPSHOTS_DIR)

        # SETUP TRAIN
        with open(self.workdir + '/train.prototxt', 'w') as f:
            f.write(str(self.model.to_train_prototxt(self.train)))
        self.solver.set_train('train.prototxt', self.train.get_count(), self.model.get_train_batch_size())

        # VAL
        for v in self.val:
            val_file = 'val-' + v.get_name() + '.prototxt'
            with open(self.workdir + '/' + val_file, 'w') as f:
                f.write(str(self.model.to_val_prototxt(v)))

            self.solver.add_val(val_file, v.get_count(), self.model.get_val_batch_size())

        with open(self.workdir + '/deploy.prototxt', 'w') as f:
            f.write(str(self.model.to_deploy_prototxt()))

        with open(self.workdir + '/solver.prototxt', 'w') as f:
            f.write(self.solver.to_solver_prototxt())

        # WRITE OR LINK MEAN IMAGE / MEAN PIXEL / INITIAL WEIGHTS
        if self.model.infmt.mean_pixel is not None:
            np.save(self.workdir + '/mean-pixel.npy', np.array(self.model.infmt.mean_pixel))

        if self.model.pretrain is not None:
            os.system('ln -s -r ' + self.model.pretrain + ' ' + self.workdir)

        # DRAW NET
        os.system('/opt/caffe/python/draw_net.py --rankdir TB {0}/train.prototxt {1}/net.png > /dev/null'
                  .format(self.workdir, self.workdir))

        # DUMP EXPERIMENT OBJ
        self.save()

    def setup_efficiently(self, experiments_dir):
        logging.info('Setting up ' + self.long_name() + ' ...')
        self.workdir = os.path.abspath(experiments_dir.rstrip('/') + '/' + self.short_name())

        mkdir_p(self.workdir + '/' + self.SNAPSHOTS_DIR)

        # COMBINED TRAIN AND VAL
        if len(self.val) != 0:
            # SETUP TRAIN-VAL with stages
            t_batch_size = self.model.get_train_batch_size()
            v_batch_size = self.model.get_val_batch_size()

            with open(self.workdir + '/train_val.prototxt', 'w') as f:
                f.write(str(self.model.to_train_val_prototxt(self.train, self.val)))
            self.solver.set_train_val('train_val.prototxt', self.train, t_batch_size, self.val, v_batch_size)

            # DRAW NET
            os.system('/opt/caffe/python/draw_net.py --rankdir TB {0}/train_val.prototxt {0}/net.png > /dev/null'
                      .format(self.workdir))

        # TRAIN-ONLY
        else:
            # SETUP TRAIN
            with open(self.workdir + '/train.prototxt', 'w') as f:
                f.write(str(self.model.to_train_prototxt(self.train)))
            self.solver.set_train('train.prototxt', self.train.get_count(), self.model.get_train_batch_size())

            # DRAW NET
            os.system('/opt/caffe/python/draw_net.py --rankdir TB {0}/train_val.prototxt {0}/net.png > /dev/null'
                      .format(self.workdir))

        with open(self.workdir + '/deploy.prototxt', 'w') as f:
            f.write(str(self.model.to_deploy_prototxt()))

        with open(self.workdir + '/solver.prototxt', 'w') as f:
            f.write(self.solver.to_solver_prototxt())

        # WRITE OR LINK MEAN IMAGE / MEAN PIXEL / INITIAL WEIGHTS
        if self.model.infmt.mean_pixel is not None:
            np.save(self.workdir + '/mean-pixel.npy', np.array(self.model.infmt.mean_pixel))

        if self.model.pretrain is not None:
            os.system('ln -s -r ' + self.model.pretrain + ' ' + self.workdir)

        # DUMP EXPERIMENT OBJ
        self.save()

    def clean(self):
        raise NotImplementedError()

    def save(self):
        with open(self.workdir + '/' + self.EXP_FILE, 'w') as f:
            pickle.dump(self, f)

    @preserve_cwd
    def run(self, plot=True, resume=False):
        os.chdir(self.workdir)

        if resume:
            last_sn = self.get_last_snapshot()
            if last_sn is None:
                resume = False
            else:
                logging.info('Resuming training from iteration {}'.format(last_sn))

        logging.info('Training on ' + self.train.get_name() + ' while validating on ' + ', '.join(
            [str(v) for v in self.val]) + ' ...')

        if os.path.exists(self.LOG_FILE) and not resume:
            os.remove(self.LOG_FILE)

        cmd = ['/opt/caffe/build/tools/caffe', 'train', '-gpu', '0', '-solver', 'solver.prototxt']

        if resume:
            cmd += ['-snapshot', '{}/snapshot_iter_{}.solverstate'.format(self.SNAPSHOTS_DIR, last_sn)]
        elif self.model.infmt.pretrain is not None:
            cmd += ['-weights', os.path.basename(self.model.infmt.pretrain)]

        caffe = subprocess.Popen(cmd, stderr=subprocess.PIPE)

        dst = subprocess.PIPE if plot else open(os.devnull, 'wb')

        tee = subprocess.Popen(['tee', '-a', self.LOG_FILE], stdin=caffe.stderr, stdout=dst)

        def handler(sig, frame):
            # propagate SIGINT down, and wait
            os.kill(caffe.pid, signal.SIGHUP)
            os.kill(caffe.pid, sig)
            caffe.wait()

        signal.signal(signal.SIGINT, handler)

        if plot:
            line_iter = iter(tee.stdout.readline, '')
            live_plot = pyffe.LivePlot(title=self.long_name())
            pyffe.LogParser(line_iter).parse(live_plot)

        tee.wait()

        # print something in case of error
        if caffe.returncode != 0:
            os.system('tail -n 20 {}'.format(self.LOG_FILE))

    # def print_test_results(self):
    #     print
    #     print self.long_name()
    #     print '==============='
    #
    #     for t in self.test:
    #         print t.get_name(), ':'
    #         test_file = self.workdir + '/test-' + t.get_name() + '.caffelog'
    #         os.system('grep "accuracy =" {} | grep -v Batch'.format(test_file))

    @preserve_cwd
    def run_test(self):
        os.chdir(self.workdir)
        if not self.test:  # no tests
            logging.info('No test defined for this experiment {}'.format(self.long_name()))
            return

        # find last snapshot
        p = re.compile('\d+')
        max_iter = str(max([int(p.findall(sn)[0]) for sn in glob.glob(self.SNAPSHOTS_DIR + '/*.caffemodel')]))

        for t in self.test:
            logging.info('Testing on ' + t.get_name() + ' ...')
            test_file = 'test-' + t.get_name() + '.prototxt'

            with open(self.workdir + '/' + test_file, 'w') as f:
                net, iters = self.model.to_test_prototxt(t)
                f.write(str(net))

            # TODO python data layer with async blob preparation
            caffe_cmd = 'caffe test -gpu 0 -model {} -weights snapshots/snapshot_iter_{}.caffemodel -iterations {} 2> test-{}.caffelog'
            os.system(caffe_cmd.format(test_file, max_iter, iters, t.get_name()))

    # FIXME the recovered snapshot differs.. I think we can blame caffe or the RNG
    #    def recover_snapshot(self, snapshot_iter):
    #        os.chdir(self.workdir)
    #        p = re.compile('\d+')
    #        snapshots = [int(p.findall(sn)[0]) for sn in glob.glob('snapshots/*.caffemodel')]
    #        prev_iter = max([s for s in snapshots if s < snapshot_iter])
    #
    #        tmp_solver = copy.copy(self.solver)
    #        tmp_solver.stop_and_snapshot_at(snapshot_iter, snapshot_iter - prev_iter)
    #        with open('tmp_solver.prototxt', 'w') as f:
    #            f.write(tmp_solver.to_solver_prototxt())
    #
    #        os.system(
    #            'caffe train -gpu 0 -solver tmp_solver.prototxt -snapshot snapshots/snapshot_iter_{}.solverstate'.format(
    #                prev_iter))

    def show_logs(self):
        plot = pyffe.LivePlot(
            title=self.long_name(),
            train=self.train,
            val=self.val
        )
        plot(self.get_log_data())

        # def summarize(self, show_train_points=True):
        #
        #     log_data = self.get_log_data()
        #     last_iter = log_data['train']['iteration'][-1]
        #     bs = log_data['meta']['batch_size'][0]
        #
        #     # list of indices where max accuracies for each test are
        #     it_idx, it_max = self.get_argmax_iters()
        #
        #     pdata = [[round(outs['accuracy'][i], 2) for i in it_idx] for k, outs in log_data['test']['out'].iteritems()]
        #     vnames = [v.get_name() for v in self.val]
        #
        #     v_idx_num = len(self.val)
        #     v_idx_names = vnames
        #
        #     if show_train_points:
        #         # XXX maybe bug in iteration/indexes? However, this method is merged with summarize_exact
        #         train_pcent = ['{0:.0f}% (~{1} imgs)'.format(100 * log_data['test']['iteration'][i] / last_iter,
        #                                                      log_data['test']['iteration'][i] * bs) for i in it_max]
        #         pdata = pdata + [train_pcent]
        #         v_idx_num = len(self.val) + 1
        #         v_idx_names = vnames + ['   --> at']
        #
        #     index = [
        #         [self.model.name] * v_idx_num,
        #         [self.train.get_name()] * v_idx_num,
        #         v_idx_names
        #     ]
        #
        #     return pd.DataFrame(pdata, index=index, columns=vnames)

        @preserve_cwd
        def trained_models_to_zip(self):
            os.chdir(self.workdir)

            _, it_max = self.get_argmax_iters()

            mname = self.model.name
            tname = self.train.get_name()

            for i, it in enumerate(it_max):
                vname = self.val[i].get_name()
                aname = "{}-on-{}-val-{}.zip".format(mname, tname, vname)
                os.system("zip -j {} deploy.prototxt".format(aname, it))
                os.system("zip -j {} {}/snapshot_iter_{}.caffemodel".format(aname, self.SNAPSHOTS_DIR, it))

    '''
    TODO DOC.
    mode='maxone' -> interesting snapshots are the max of some validation set
    mode='maxmin' -> interesting snapshot is the one giving the max of the min accuracy on val sets
    mode='maxavg' -> interesing snapshot is the one giving the max average of accuracy on val sets
    '''

    @preserve_cwd
    def summarize(self, datasets=None, exact=True, show_train_points=True, mode='maxone'):
        assert mode in ['maxone', 'maxmin', 'maxavg'], "mode is not one of ('maxone', 'maxmin', 'maxavg'): %r" % mode

        if datasets is None:
            datasets = self.test

        logging.debug('Summarizing {} ...'.format(self.short_name()))
        # TODO: caching the DataFrame on CSV on disk.
        # os.chdir(self.workdir)
        # if os.path.exists('exact_summary.csv'):
        #    return pd.read_csv('exact_summary.csv')

        log_data = self.get_log_data()
        last_iter = log_data['train']['iteration'][-1]
        bs = log_data['meta']['batch_size'][0]

        # list of indices where wanted accuracies for each test are        
        if mode is 'maxone':
            it_idx, it_max = self.get_argmax_iters()
            vnames = [v.get_name() for v in self.val]
        elif mode is 'maxmin':
            i, j = self.get_max_min_val_iter()
            it_idx = [i]
            it_max = [j]
            vnames = ['maxmin of' + [v.get_name() for v in self.val].join(',')]
        elif mode is 'maxavg':
            i, j = self.get_max_avg_val_iter()
            it_idx = [i]
            it_max = [j]
            vnames = ['maxavg of' + [v.get_name() for v in self.val].join(',')]

        if exact:
            pdata = [[round(self.get_accuracy_at(it, v), 4) for it in it_max] for v in datasets]
        else:
            pdata = [[round(outs['accuracy'][i], 4) for i in it_idx] for k, outs in log_data['test']['out'].iteritems()]

        dnames = [v.get_name() for v in datasets]

        d_idx_num = len(datasets)
        d_idx_names = dnames

        if show_train_points:
            train_pcent = ['{0:.0f}% (~{1} imgs)'.format(100.0 * it / last_iter, it * bs) for it in it_max]
            pdata = pdata + [train_pcent]
            d_idx_num = len(datasets) + 1
            d_idx_names = dnames + ['   --> at']

        index = [
            [self.model.name] * d_idx_num,
            [self.train.get_name()] * d_idx_num,
            d_idx_names
        ]

        df = pd.DataFrame(pdata, index=index, columns=vnames)
        # df.to_csv('exact_summary_last.csv')
        return df
