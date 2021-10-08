import os
import csv
import copy
import datetime as dt
import random

from typing import List, Any

from pyoperant.behavior import shape, two_alt_choice
from pyoperant.errors import EndSession, EndBlock
from pyoperant import components, utils, reinf, queues
import numpy
from pyoperant.reinf import FixedRatioSchedule


class TestFRSchedule(FixedRatioSchedule):
    """Fixed ratio reinforcement schedule with test stims
    """

    def __init__(self, reinforce_test=True, ratio=1):
        super(TestFRSchedule, self).__init__(ratio=ratio)  # this runs _update
        self.reinforce_test = reinforce_test
        self.inject_test = True  # inject test on the first trial

    def _update(self):
        self.cumulative_correct = 0
        self.threshold = self.ratio
        # TODO add option to do more than 1 test trials in a row

    def consequate(self, trial):
        """Consequate the trial response. Correction trials do not go through this func
        """
        assert hasattr(trial, 'assigned_correctness') and \
               isinstance(trial.assigned_correctness, bool)

        if trial.assigned_correctness:
            self.cumulative_correct += 1

            # if over threshold, reward, VR ends, inject test next
            if self.cumulative_correct >= self.threshold:
                self.cumulative_correct = 0
                # if test trial and not reinforced, do normal trial next, shorter streak
                if trial.type_ == 'test' and not self.reinforce_test:
                    self.threshold = random.randint(1, 2 * self.ratio - 1)
                    self.inject_test = False
                    return False
                # else, inject test trial next
                else:
                    self.threshold = random.randint(1, 2 * self.ratio)
                    self.inject_test = True
                    return True
            # else no reward, normal trial next
            else:
                self.inject_test = False
                return False

        elif trial.assigned_correctness == False:
            self._update()  # reset the threshold, cum_correct

            # if this is a test trial, no test injection next, punish based on setting
            if trial.type_ == 'test':
                # update the threshold so that the bird doesn't have to do a longer streak
                self.threshold = random.randint(1, 2 * self.ratio-1)
                self.inject_test = False
                if not self.reinforce_test:
                    return False
                else:
                    return True

            # else punish, start new VR, inject test next
            else:
                self.inject_test = True
                return True
        else:  # what is there to else?
            return False


class TestVRSchedule(TestFRSchedule):
    """Variable ratio schedule with test stims

    """

    def __init__(self, reinforce_test, ratio=1):
        super(TestVRSchedule, self).__init__(reinforce_test=reinforce_test, ratio=ratio)

    def _update(self):
        """ update min correct by randomly sampling from interval [1:2*ratio)"""
        self.cumulative_correct = 0
        self.threshold = random.randint(1, 2 * self.ratio)

    def __unicode__(self):
        return "VR%i" % self.ratio


class TwoACTestExp(two_alt_choice.TwoAltChoiceExp):
    """A two alternative choice experiment with test stims mixed in. Must be used with variable ratio.


    Parameters
    ----------


    Attributes
    ----------
    req_panel_attr : list
        list of the panel attributes that are required for this behavior
    fields_to_save : list
        list of the fields of the Trial object that will be saved
    trials : list
        all of the trials that have run
    shaper : Shaper
        the protocol for shaping
    parameters : dict
        all additional parameters for the experiment
    data_csv : string
        path to csv file to save data
    reinf_sched : object
        does logic on reinforcement



    """

    def __init__(self, *args, **kwargs):
        super(TwoACTestExp, self).__init__(*args, **kwargs)
        assert 'test_stims' in self.parameters.keys()
        assert 'with_test' in self.parameters.keys()
        self.with_test = self.parameters['with_test']

        if self.with_test:
            # if empty test stims then construct the test stim set from motifs
            if self.parameters['test_stims'] is None:
                assert self.parameters['n_test_motifs'] > 0
                # TODO ADD MOTIFS TO TEST STIMS
            else:
                self.test_stims = self.parameters['test_stims']

        assert 'reinforcement' in self.parameters.keys()

        self.reinforcement= self.parameters['reinforcement']
        if self.reinforcement['schedule'] == 'variable_ratio':
            self.reinf_sched = TestVRSchedule(reinforce_test=self.reinforcement['test'],
                                              ratio=self.reinforcement['ratio'])
        elif self.reinforcement['schedule'] == 'fixed_ratio':
            self.reinf_sched = TestFRSchedule(reinforce_test=self.reinforcement['test'],
                                              ratio=self.reinforcement['ratio'])
        else:
            raise ValueError('Unknown reinforcement schedule')

    def session_main(self):
        """ Runs the sessions

        Inside of `session_main`, we loop through sessions and through the trials within
        them. This relies heavily on the 'block_design' parameter, which controls trial
        conditions and the selection of queues to generate trial conditions.

        """

        def run_trial_queue():
            for tr_cond in self.trial_q:
                try:
                    # inject test when a VR ends, or when there is punishment
                    consecutive_tests = 0
                    while self.reinf_sched.inject_test:
                        # if too many test trials are run consecutively due to lack of
                        # responses, move on to normal trials
                        if consecutive_tests >= 3:
                            break
                        consecutive_tests += 1
                        self.new_trial(trial_type='test')
                        self.run_trial()
                    self.new_trial(tr_cond)
                    self.run_trial()
                    while self.do_correction:
                        self.new_trial(tr_cond)
                        self.run_trial()
                except EndBlock:
                    self.trial_q = None
                    break
            self.trial_q = None

        if self.session_q is None:
            self.log.info('Next sessions: %s' % self.parameters['block_design']['order'])
            self.session_q = queues.block_queue(self.parameters['block_design']['order'])

        if self.trial_q is None:
            for sn_cond in self.session_q:  # sn_cond is default to be "default"

                self.trials = []
                self.do_correction = False
                self.inject_test = True
                self.session_id += 1
                self.log.info('starting session %s: %s' % (self.session_id, sn_cond))

                # grab the block details
                blk = copy.deepcopy(self.parameters['block_design']['blocks'][sn_cond])

                # load the block details into the trial queue
                q_type = blk.pop('queue')
                if q_type == 'random':
                    self.trial_q = queues.random_queue(**blk)
                elif q_type == 'block':
                    self.trial_q = queues.block_queue(**blk)
                elif q_type == 'mixedDblStaircase':
                    dbl_staircases = [queues.DoubleStaircaseReinforced(stims) for stims in blk['stim_lists']]
                    self.trial_q = queues.MixedAdaptiveQueue.load(
                        os.path.join(self.parameters['experiment_path'], 'persistentQ.pkl'), dbl_staircases)
                try:
                    run_trial_queue()
                except EndSession:
                    return 'post'

            self.session_q = None

        else:
            self.log.info('continuing last session')
            try:
                run_trial_queue()
            except EndSession:
                return 'post'

        return 'post'

    ## trial flow
    def new_trial(self, conditions=None, trial_type='normal'):
        """Creates a new trial and appends it to the trial list

        If `self.do_correction` is `True`, then the conditions are ignored and a new
        trial is created which copies the conditions of the last trial.

        Parameters
        ----------
        conditions : dict
            The conditions dict must have a 'class' key, which specifys the trial
            class. The entire dict is passed to `exp.get_stimuli()` as keyword
            arguments and saved to the trial annotations.

        trial_type: str
            Trial type. Default to be "normal", other options are "correction" and "test"

        """
        if len(self.trials) > 0:
            index = self.trials[-1].index + 1
        else:
            index = 0

        if self.do_correction:
            trial_type = 'correction'

        if not conditions:
            # if no conditions specified for test trials, randomly select conditions
            if trial_type == 'test':
                all_test_stims = sorted(self.test_stims.keys())  # type: List
                test_index = random.randrange(len(all_test_stims))
                conditions = self.test_stims[all_test_stims[test_index]]
            else:
                raise ValueError

        if trial_type == 'correction':
            # for correction trials, we want to use the last trial as a template
            trial = utils.Trial(type_=trial_type,
                                index=index,
                                class_=self.trials[-1].class_)
            for ev in self.trials[-1].events:
                if ev.label is 'wav':
                    trial.events.append(copy.copy(ev))
                    trial.stimulus_event = trial.events[-1]
                    trial.stimulus = trial.stimulus_event.name
                elif ev.label is 'motif':
                    trial.events.append(copy.copy(ev))
            self.log.debug("correction trial: class is %s" % trial.class_)
        else:
            # otherwise, we'll create a new trial (test or normal)
            trial = utils.Trial(index=index,
                                type_=trial_type)
            trial.class_ = conditions['class']
            trial_stim, trial_motifs = self.get_stimuli(**conditions)
            trial.events.append(trial_stim)
            trial.stimulus_event = trial.events[-1]
            trial.stimulus = trial.stimulus_event.name
            for mot in trial_motifs:
                trial.events.append(mot)

        trial.session = self.session_id
        trial.annotate(**conditions)

        self.trials.append(trial)
        self.this_trial = self.trials[-1]
        self.this_trial_index = self.trials.index(self.this_trial)
        self.log.debug("trial %i: %s, %s" % (self.this_trial.index, self.this_trial.type_, self.this_trial.class_))

        return True

    def get_stimuli(self, **conditions):
        """ Get the trial's stimuli from the conditions

        Returns
        -------
        stim, epochs : Event, list


        """
        stim_name = conditions['stim_name']
        stim_file = self.parameters['stims'][stim_name]  # TODO either integrate test stim or code new func
        self.log.debug(stim_file)

        stim = utils.auditory_stim_from_wav(stim_file)
        epochs = []
        return stim, epochs

    def trial_post(self):
        '''things to do at the end of a trial'''
        self.this_trial.duration = (dt.datetime.now() - self.this_trial.time).total_seconds()
        self.analyze_trial()
        self.save_trial(self.this_trial)
        self.write_summary()
        utils.wait(self.parameters['intertrial_min'])

        # determine if next trial should be a correction trial
        self.do_correction = True
        if len(self.trials) > 0:
            if self.parameters['correction_trials']:
                # correct trials avoid correction
                if self.this_trial.correct == True:
                    self.do_correction = False
                # no response
                elif self.this_trial.response == 'none':
                    # normal trials without response do correction based on setting
                    if self.this_trial.type_ == 'normal':
                        self.do_correction = self.parameters['no_response_correction_trials']
                    # test trials without responses do not need to be corrected
                    elif self.this_trial.type_ == 'test':
                        self.do_correction = False
                    # correction trials without responses need to be corrected again
                    else:
                        self.do_correction = True
                # wrong trials
                else:
                    # test trials do not need corrections
                    if self.this_trial.type_ == 'test':
                        self.do_correction = False
                    else:
                        self.do_correction = True
            else:
                self.do_correction = False
        else:
            # why would there be 0 trials?
            self.do_correction = False

        if self.check_session_schedule() == False:
            raise EndSession
        if self._check_free_food_block(): return 'free_food_block'

    def consequence_main(self):

        if self.this_trial.response == self.parameters['classes'][self.this_trial.class_]['component']:
            self.this_trial.correct = True
        elif self.this_trial.response == 'none':
            # if the bird ignores test stim, inject another test next
            if self.this_trial.type_ == 'test':
                self.reinf_sched.inject_test = True
            pass
        else:
            self.this_trial.correct = False

        # if test trial on random reinforcement, assign a correctness value
        if self.this_trial.type_ == 'test' and self.reinforcement['test'] == 'random':
            self.this_trial.assigned_correctness = bool(random.getrandbits(1))
        # else use the real value
        else:
            self.this_trial.assigned_correctness = self.this_trial.correct

        # reward
        if self.this_trial.assigned_correctness:

            # secondary reinforcement
            if self.reinforcement['secondary']:
                secondary_reinf_event = self.secondary_reinforcement()
                # self.this_trial.events.append(secondary_reinf_event)

            # if correction trial, meaning starting a new VR, inject test
            if self.this_trial.type_ == 'correction':
                self._run_correction_reward()

            # otherwise reward based on consequate
            # inject test because if rewarded it means a new VR starts
            elif self.reinf_sched.consequate(trial=self.this_trial):
                self.reward_pre()
                self.reward_main()  # provide a reward
                self.reward_post()

            # correct but not rewarding
            # could be 1) middle of VR, 2) random reinforcement
            else:
                pass

        # punish
        else:
            if self.reinf_sched.consequate(trial=self.this_trial):
                self.punish_pre()
                self.punish_main()
                self.punish_post()

    # TODO integrate test stim into queue system

    def _run_correction_reward(self):
        # added secondary reinforcement to correction trials
        secondary_reinf = self.secondary_reinforcement()