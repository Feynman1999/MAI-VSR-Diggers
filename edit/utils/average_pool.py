from abc import ABC
from collections import defaultdict


class AVERAGE_POOL(ABC):
    """
        support type:
            key : float/int
            key : list of float/int
    """
    def __init__(self, average_length=10):
        assert average_length >= 1 and isinstance(average_length, int)
        assert average_length <= 1000 , 'average length should not too large'
        self.average_length = average_length
        self.pool = defaultdict(list)

    def _get(self, key):
        """
            return  average  list  or  float
        """
        if self.pool[(key, 0)] != []: # need to return list
            res = []
            for idx in range(100):
                List = self.pool[(key, idx)]
                if List == []:
                    break
                res.append(sum(List) / len(List))
            return res
        else:
            List = self.pool[key]
            return sum(List) / len(List)
            
    def _maintain(self, List, val):
        List.append(val)
        if len(List) > self.average_length:
            List.pop(0) # remove the first one

    def update(self, key, value):
        if isinstance(value, list):
            for (idx, v) in enumerate(value):
                assert idx < 100, "the list which need to do average is too long!"
                List = self.pool[(key, idx)]
                self._maintain(List, v)
        else:
            List = self.pool[key]
            self._maintain(List, value)

        return self._get(key)