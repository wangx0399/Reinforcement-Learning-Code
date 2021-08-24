import numpy as np

# one by one given data, and iterative compute the mean and variance
# then to normalization


class SlidingMandS(object):
    """
    when add a new data, recompute the whole data mean and variance
    """
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)         # mean
        self._S = np.zeros(shape)         # total variance

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class Normalization:
    """
    y = (x-mean)/std
    using running estimates of mean,std by  ***SlidingMS***
    """
    def __init__(self, shape, domean=True, dostd=True, clip=10.0):
        self.domean = domean
        self.dostd = dostd
        self.clip = clip

        self.rs = SlidingMandS(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.domean:
            x = x - self.rs.mean
        if self.dostd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape




# another code for the same normalization

class State_norm:
    def __init__(self, s_dim, s_clip=5.0):
        self.pointer = 0
        self.mean = np.zeros(s_dim, dtype=np.float32)
        self.nvar = np.zeros(s_dim, dtype=np.float32)         # nvar = n * var
        self.var = np.zeros(s_dim, dtype=np.float32)
        self.std = np.zeros(s_dim, dtype=np.float32)
        self.clip = s_clip

    def mean_std(self, s):
        s = np.asarray(s, dtype=np.float32)
        assert s.shape == self.mean.shape
        self.pointer += 1
        if self.pointer == 1:
            self.mean = s                                     # self.std = [0, 0, 0]
        else:
            old_mean = self.mean
            self.mean = old_mean + (s - old_mean) / self.pointer
            self.nvar = self.nvar + (s - old_mean) * (s - self.mean)
            self.var = self.nvar / (self.pointer -1)
            self.std = np.sqrt(self.var)
        return self.mean, self.std

    def normalize(self, s, isclip=True):
        mean, std = self.mean_std(s)
        if std.all() == 0:
            s = s
        else:
            s = (s - mean) / (std + 1e-6)
        if isclip:
            s = np.clip(s, -self.clip, self.clip)
        return s
