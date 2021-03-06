"""
    Taken from https://raw.githubusercontent.com/openai/baselines/master/baselines/logger.py
    Slightly modified.
"""



import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
from collections import defaultdict
from torch_rl.config import tensorboard_path, logging_path, benchmark_path, video_path, root_path


LOG_OUTPUT_FORMATS = ['stdout', 'log', 'csv']
# Also valid: json, tensorboard

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 5

levels = [DEBUG, INFO, WARN, ERROR, DISABLED]



def level_check(func):
    def dec(*args, **kwargs):
        if 'level' in kwargs:
            level = kwargs['level']
            assert level in levels, "Unknown logging level"
        return func(*args, **kwargs)
    return dec 
 
class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError

class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError

class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'read'), 'expected file or str, got %s'%filename_or_file
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print('WARNING: tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s

    def writeseq(self, seq):
        for arg in seq:
            self.file.write(arg)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()

class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, 'wt')

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, 'dtype'):
                v = v.tolist()
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()

class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, 'w+t')
        self.keys = []
        self.sep = ','

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = kvs.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(k)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """
    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = 'events'
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat
        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {'tag': k, 'simple_value': float(v)}
            return self.tf.Summary.Value(**kwargs)
        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = self.step # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None

def make_output_format(format, ev_dir, log_suffix=''):
    if format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif format == 'log':
        ev_dir = logging_path()
        os.makedirs(ev_dir, exist_ok=True)
        return HumanOutputFormat(osp.join(ev_dir, 'log%s.txt' % log_suffix))
    elif format == 'json':
        ev_dir = benchmark_path()
        os.makedirs(ev_dir, exist_ok=True)
        return JSONOutputFormat(osp.join(ev_dir, 'progress%s.json' % log_suffix))
    elif format == 'csv':
        ev_dir = benchmark_path()
        os.makedirs(ev_dir, exist_ok=True)
        return CSVOutputFormat(osp.join(ev_dir, 'progress%s.csv' % log_suffix))
    elif format == 'tensorboard':
        ev_dir = tensorboard_path()
        os.makedirs(ev_dir, exist_ok=True)
        return TensorBoardOutputFormat(osp.join(ev_dir,  ev_dir + '%s' % log_suffix))
    else:
        raise ValueError('Unknown format specified: %s' % (format,))

# ================================================================
# API
# ================================================================

def logkv(key, val, level=INFO):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    Logger.CURRENT.logkv(key, val, level)

def logkv_mean(key, val, level=INFO):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    Logger.CURRENT.logkv_mean(key, val, level)

def logkvs(d, level=INFO):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v, level)

def dumpkvs():
    """
    Write all of the diagnostics from the current iteration

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    Logger.CURRENT.dumpkvs()


def printkvs():
    """
    Write all of the diagnostics from the current iteration, but don't clear

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    Logger.CURRENT.printkvs()


def getkvs():
    return Logger.CURRENT.name2val


def __get__(key):
    return Logger.CURRENT.name2val[key]


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    Logger.CURRENT.log(*args, level=level)

def debug(*args):
    log(*args, level=DEBUG)

def info(*args):
    log(*args, level=INFO)

def warn(*args):
    log(*args, level=WARN)

def error(*args):
    log(*args, level=ERROR)

@level_check
def set_level(level):
    """
    Set logging threshold on current logger.
    """
    Logger.CURRENT.set_level(level)

def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return Logger.CURRENT.get_dir()

record_tabular = logkv
dump_tabular = dumpkvs

class ProfileKV:
    """
    Usage:
    with logger.ProfileKV("interesting_scope"):
        code
    """
    def __init__(self, n):
        self.n = "wait_" + n
    def __enter__(self):
        self.t1 = time.time()
    def __exit__(self ,type, value, traceback):
        Logger.CURRENT.name2val[self.n] += time.time() - self.t1

def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """
    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with ProfileKV(n):
                return func(*args, **kwargs)
        return func_wrapper
    return decorator_with_name


# ================================================================
# Backend
# ================================================================

class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
                    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, clear=True):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.clear_dicts = clear

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val, level=INFO):
        if self.level <= level:
            self.name2val[key] = val

    def logkv_mean(self, key, val, level=INFO):
        if self.level > level:
            return
        if val is None:
            self.name2val[key] = None
            return
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval*cnt/(cnt+1) + val/(cnt+1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        """
            Prints to output and clears the buffer.
        """
        self.printkvs()
        if self.clear_dicts:
            self.clear()

    def printkvs(self):
        """
            Just prints to output, doesn't clear.
        """
        if self.level == DISABLED: return
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(self.name2val)

    def clear(self):
        self.name2val.clear()
        self.name2cnt.clear()

    def dec(key, num=1):
        self.name2val.get(key, 0)
        self.name2val[key] = val - num

    def inc(key, num=1):
        val = self.name2val.get(key, 0)
        self.name2val[key] = val + num

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))

Logger.DEFAULT = Logger.CURRENT = Logger(dir=None, output_formats=[HumanOutputFormat(sys.stdout)])

def configure(dir=None, output_formats=None, clear=True):
    if dir is None:
        dir = os.getenv('TRL_LOG_DIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("torch-rl-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    if output_formats is None:
        strs = os.getenv('TRL_LOG_FORMAT')
        output_formats = strs.split(',') if strs else LOG_OUTPUT_FORMATS
    output_formats = [make_output_format(f, dir) for f in output_formats]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, clear=clear)
    log('Logging to %s'%dir)

def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log('Reset logger')

class scoped_configure(object):
    def __init__(self, dir=None, format_strs=None):
        self.dir = dir
        self.format_strs = format_strs
        self.prevlogger = None
    def __enter__(self):
        self.prevlogger = Logger.CURRENT
        configure(dir=self.dir, format_strs=self.format_strs)
    def __exit__(self, *args):
        Logger.CURRENT.close()
        Logger.CURRENT = self.prevlogger

# ================================================================

def _demo():
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    dir = "/tmp/testlogging"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    configure(dir=dir)
    logkv("a", 3)
    logkv("b", 2.5)
    dumpkvs()
    logkv("b", -2.5)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see a = 5.5")
    logkv_mean("b", -22.5)
    logkv_mean("b", -44.4)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see b = 33.3")

    logkv("b", -2.5)
    dumpkvs()

    logkv("a", "longasslongasslongasslongasslongasslongassvalue")
    dumpkvs()



def _factory_demo():
    factory = LoggerFactory()
    logger1 = factory.create_logger()
    logger1.logkv("a",  6)

    factory.dump()

# ================================================================
# Readers
# ================================================================

def read_json(fname):
    import pandas
    ds = []
    with open(fname, 'rt') as fh:
        for line in fh:
            ds.append(json.loads(line))
    return pandas.DataFrame(ds)

def read_csv(fname):
    import pandas
    return pandas.read_csv(fname, index_col=None, comment='#')

def read_tb(path):
    """
    path : a tensorboard file OR a directory, where we will find all TB files
           of the form events.*
    """
    import pandas
    import numpy as np
    from glob import glob
    from collections import defaultdict
    import tensorflow as tf
    if osp.isdir(path):
        fnames = glob(osp.join(path, "events.*"))
    elif osp.basename(path).startswith("events."):
        fnames = [path]
    else:
        raise NotImplementedError("Expected tensorboard file or directory containing them. Got %s"%path)
    tag2pairs = defaultdict(list)
    maxstep = 0
    for fname in fnames:
        for summary in tf.train.summary_iterator(fname):
            if summary.step > 0:
                for v in summary.summary.value:
                    pair = (summary.step, v.simple_value)
                    tag2pairs[v.tag].append(pair)
                maxstep = max(summary.step, maxstep)
    data = np.empty((maxstep, len(tag2pairs)))
    data[:] = np.nan
    tags = sorted(tag2pairs.keys())
    for (colidx,tag) in enumerate(tags):
        pairs = tag2pairs[tag]
        for (step, value) in pairs:
            data[step-1, colidx] = value
    return pandas.DataFrame(data, columns=tags)


class LoggerFactory(object):
    """
        A class that creates loggers and keeps track of them,
        this is for convenience when we have many loggers on
        different levels tracking differt things at the same time.
    """
    def __init__(self):
        super(LoggerFactory, self).__init__()

        self.loggers = []

    @level_check
    def create_logger(self, dir=None, output_formats=[HumanOutputFormat(sys.stdout)], level=INFO):
        """
            Creates a logger and observes it.
        """
        logger = Logger(dir, output_formats)
        logger.set_level(level)
                
        self.loggers.append(logger)

        return logger

    def dump(self):
        for logger in self.loggers:
            logger.dumpkvs()

    def close(self):
        """
            Closes alls loggers
        """
        for logger in self.loggers:
            logger.close() 




def save_params(**kwargs):
    import json
    dir = root_path()
    with open(os.path.join(dir, 'params.json'), 'w') as f:
        json.dump(kwargs, f, indent=4)





if __name__ == "__main__":
    _demo()
    _factory_demo()