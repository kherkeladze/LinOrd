import random
import numpy as np
import pandas as pd

from psychopy import visual, event, core

# TODOs:
# - [ ] copy time transformation
# - [ ] copy fixation object
# - [ ] copy loading settings etc.
# - [ ] copy marker sending from my proc
# - [ ] 

# out df:
# model   show_order  condition  question question_type
# 'ABCD'  1,2,3,4,2,3 'easy'     'CA'     2             0
# correct_answer  answer ifcorrect RT
# 0               1      0         1.2189

class LinOrdExperiment(object):
	def __init__(self, window, config_file):
		# super class should do this:
		self.window = window
		self.times = settings['times']
		# turn to frames

		self.letters = list('bcdfghjklmnprstwxz')
		self.relations = ['>', '<']

		self.create_stimuli()

		# wszystkie możliwe sekwencje
		# w wierszach warunki, w kolumnach wersje
		self.conditions = np.array( [[[1, 2, 2, 3, 3, 4],
			[3, 4, 2, 3, 1, 2]], [[2, 3, 1, 2, 3, 4],
			[2, 3, 3, 4, 1, 2]], [[1, 2, 3, 4, 2, 3],
			[3, 4, 1, 2, 2, 3]]] ) - 1

	def get_time(self, stim):
		time = self.times[stim]
		if isinstance(time, list):
			return random.randint(**time)
		else
			return time

	def create_stimuli(self):
		args = {'units': 'deg', 'height': self.settings['letter_height']}
		self.stim = {l: visual.TextStim(self.window, text=l, **args)
			for l in self.letters + self.relations}
		if self.settings['fix_between_pairs']:
			self.stim.update(btw_pairs=self.stim['fix'])
		if self.settings['fix_during_wait']:
			self.stim.update(dur_wait=self.stim['fix'])

	def _show_pair(self, pair, times):
		# show_pair can return randomized times
		elems = pair.split('')
		[self.show_element(el, tm) for el, tm in zip(elems, times)]

	def show_element(self, elem, time):
		if elem not in self.stim:
			elem = False
		# draw element
		for f in range(time):
			if elem:
				self.stim[elem].draw()
			self.window.flip()

	def show_pair(self, pair):
		events = ['element', 'before_relation', 'relation',
			'after_relation', 'element']
		times = map(self.get_time, events)
		self._show_pair(pair, times)
		return times

	def ask_question(self, question):
		# show relation
		# wait for response
		# return response

	def show_premises(self, model, sequence, relation):
		all_times = list()
		for i in [[0,1], [2,3], [4,5]]:
			pair = model[sequence[i]]
			pair = ' '.join(pair[0], relation, pair[1])
			times = self.show_pair(pair)
			if i is not [4,5]:
				next_time = self.get_time('after_pair')
				self.show_element('btw_pairs', next_time)
			else:
				next_time = self.get_time('after_last_pair')
				self.show_element('dur_wait', next_time)
			# add to times and append to all_times
			times += [next_time]
			all_times.append(times)
		return np.array(times)

	def show_trial(self, trial, ):
		pass

	def create_trials(self):
		sequence = random.sample(self.letters, 4)
		sequence = np.array(sequence)

		# losujemy warunek i wersję
		ind = random.sample(range(4), 1) + random.sample(range(2), 1)
		current_cond = conditions[ind[0], ind[1], :]

# wypisz kolejne pary
relation = random.sample(['<', '>'], 1)[0]
for i in [[0,1], [2,3], [4,5]]:
	pair = sequence[current_cond[i]]
	print(pair[0], relation, pair[1])