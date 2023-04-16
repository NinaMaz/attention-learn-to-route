import torch
from collections import defaultdict



@torch.jit.script
def Q(values: torch.Tensor, rewards: torch.Tensor, not_done: torch.Tensor, gamma: float):
    next_values = torch.cat([
        (values.detach() * not_done)[1:],
        torch.zeros([1]+values.shape[1:], device=values.device)])
    qvalues = rewards + gamma * next_values
    return qvalues * not_done



@torch.jit.script
def A(values: torch.Tensor, rewards: torch.Tensor, not_done: torch.Tensor, gamma: float):
    qvalues = Q(values, rewards, not_done, gamma)
    advantage = (qvalues - values) * not_done
    return advantage



class Trajectory:
    def __init__(self):
        self._data = defaultdict(list)

    def __getitem__(self, k):
        if isinstance(k, tuple):
            return tuple(self.__getitem__(key) for key in k)
        if k not in self._data:
            raise KeyError(f"Key '{k}' not found in trajectory")
        return torch.stack(self._data[k])

    def __getattr__(self, key):
        if key in self._data:
            return self[key]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def keys(self):
        return self._data.keys()

    def append(self, key, value):
        if isinstance(value, dict):
            value = TensorDict(value)
        self._data[key].append(value.clone())



class Sampler:
    """
    Samples minibatches from trajectory for a number of epochs.
    All data should be with shape [length, batch_size, ...]
    """
    def __init__(self, minibatch_size, data, LxB):
        self.LxB = LxB
        self.stored_data = [d.flatten(end_dim=1) for d in data]
        self.batch_size = minibatch_size

    def get_next(self):
        """ Returns next minibatch. """
        idx = torch.randperm(self.LxB)
        return (d[idx[:self.batch_size]] for d in self.stored_data)



class TensorDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k,v in self.items():
            super().__setitem__(k, v.clone())

    def __repr__(self):
        return f"TensorDict({dict(self)})"

    @staticmethod
    def convert(new):
        if isinstance(next(iter(new.values())), torch.Tensor):
            return TensorDict(new)
        else:
            return new

    def __getitem__(self, item):
        if isinstance(item, str):
            return super().__getitem__(item)
        return TensorDict({k : x.__getitem__(item) for k, x in self.items()})

    def __getattr__(self, name):
        attrs = [getattr(v, name) for v in self.values()]

        if callable(getattr(torch.Tensor, name)):
            def func(*args, **kwargs):
                new = {k : f(*args, **kwargs) for k, f in zip(self.keys(), attrs)}
                return TensorDict.convert(new)
            return func
        else:
            return TensorDict.convert({k : f for k, f in zip(self.keys(), attrs)})

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if (not all(issubclass(t, TensorDict) for t in types)
                or not func in [torch.cat, torch.stack]):
            return NotImplemented
        td = next(iter(next(iter(args))))
        ttypes = [torch.Tensor] * len(args)
        new = {k : torch.Tensor.__torch_function__(func, ttypes, [[x[k] for x in a] for a in args], kwargs)
               for k in td.keys()}
        return TensorDict.convert(new)