

class BaseSchedule(object):
    """Maintains logic for deciding whether to consequate trials.

    This base class provides the most basic reinforcent schedule: every
    response is consequated.

    Methods:
    consequate(trial) -- returns a boolean value based on whether the trial
        should be consequated. Always returns True.

    """
    def __init__(self):
        super(BaseSchedule, self).__init__()

    def consequate(self,trial):
        if trial.correct:
            return True
        else:
            return True

class ContinuousReinforcement(BaseSchedule):
    """Maintains logic for deciding whether to consequate trials.

    This base class provides the most basic reinforcent schedule: every
    response is consequated.

    Methods:
    consequate(trial) -- returns a boolean value based on whether the trial
        should be consequated. Always returns True.

    """
    def __init__(self):
        super(ContinuousReinforcement, self).__init__()

    def consequate(self,trial):
        if trial.correct:
            return True
        else:
            return True

class FixedRatioSchedule(ReinforcementSchedule):
    """Maintains logic for deciding whether to consequate trials.

    This class implements a fixed ratio schedule, where a reward reinforcement 
    is provided after every nth correct response, where 'n' is the 'ratio'.

    Incorrect trials are always reinforced.

    Methods:
    consequate(trial) -- returns a boolean value based on whether the trial
        should be consequated.

    """
    def __init__(self, ratio=1):
        super(FixedSchedule, self).__init__()
        self.ratio = ratio
        self.cumulative_correct = 0
        self._update()

    def _update(self):
        self.min_correct = ratio

    def consequate(self,trial):
        if trial.correct:
            self.cumulative_correct += 1
            if self.cumulative_correct >= self.min_correct:
                self._update()
                return True
            else:
                return False
        else:
            self.cumulative_correct = 0
            return True

    def __unicode__(self):
        return "FR%i" % self.ratio

class VariableRatioSchedule(FixedRatioSchedule):
    """Maintains logic for deciding whether to consequate trials.

    This class implements a variable ratio schedule, where a reward 
    reinforcement is provided after every a number of consecutive correct 
    responses. On average, the number of consecutive responses necessary is the
    'ratio'. After a reinforcement is provided, the number of consecutive 
    correct trials needed for the next reinforcement is selected by sampling 
    randomly from the interval [1,2*ratio-1]. e.g. a ratio of '3' will require 
    consecutive correct trials of 1, 2, 3, 4, & 5, randomly.

    Incorrect trials are always reinforced.

    Methods:
    consequate(trial) -- returns a boolean value based on whether the trial
        should be consequated.


    """
    def __init__(self, ratio=1):
        super(VariableRatioSchedule, self).__init__(ratio=ratio)

    def _update(self):
        ''' update min correct by randomly sampling from interval [1:2*ratio-1)'''
        self.min_correct = random.randint(1, 2*self.ratio-1)

    def __unicode__(self):
        return "VR%i" % self.ratio
        