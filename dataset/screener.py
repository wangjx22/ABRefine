"""Code."""
import random

class BaseScreener(object):
    """Define Class BaseScreener."""

    def __init__(self):
        """Run __init__ method."""
        # code.
        super(BaseScreener, self).__init__()

    def screen(self, value):
        """Run screen method."""
        # code.
        raise NotImplementedError(f"screen method not implemented.")



class UpperboundScreener(BaseScreener):
    """Define Class UpperboundScreener."""

    def __init__(self, upper_bound, target):
        """Run __init__ method."""
        # code.
        super(UpperboundScreener, self).__init__()
        self.upper_bound = upper_bound
        self.target = target

    def screen(self, value):
        """Run screen method."""
        # code.
        if value < self.upper_bound:
            return True
        else:
            return False

class ProbabilityScreener1d(BaseScreener):
    """Define Class ProbabilityScreener1d."""

    def __init__(self, probs, target):
        """Run __init__ method."""
        # code.
        super(ProbabilityScreener1d, self).__init__()
        self.probs = probs              # a dict {key:value}, indicates P(key[i-1]<x<key[i])=value[i]; the last key should be 1e9
        self.target = target
        self.keys = list(self.probs.keys())

    def screen(self, value):
        """Run screen method."""
        # code.
        thres = None
        for i in range(len(self.keys)):
            if value < self.keys[i]:
                thres = self.probs[self.keys[i]]
                break

        rnd = random.random()
        if rnd < thres:
            return True
        else:
            return False
