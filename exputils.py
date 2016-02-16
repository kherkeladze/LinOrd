# -*- encoding: utf-8 -*-
from __future__ import division, print_function

import os
import six
import yaml
import random
import numpy as np
import pandas as pd

from psychopy import visual, event, core

# TODOs:
# - [x] test przerwa
# - [x] add info on resp buttons to przerwa
# - [x] save output
# - [x] fill_df
# - [x] save_responses
# - [ ] copy marker sending from my proc
# - [ ] add feedback to questions
# - [ ] add save/nosave option


class LinOrdExperiment(object):
	def __init__(self, window, config_file):
		# super class should do this:
		self.window = window
		self.frame_time = get_frame_time(window)

		# load settings
		file_name = os.path.join(os.getcwd(), config_file)
		with open(file_name, 'r') as f:
		    settings = yaml.load(f)
	
		self.times = s2frames(settings['times'], self.frame_time)
		self.times['response'] = settings['times']['response']
		self.settings = settings

		self.resp_keys = settings['resp_keys']
		rnd = random.sample([True, False], 1)[0]
		self.resp_mapping = {self.resp_keys[0]: rnd}
		self.resp_mapping.update({self.resp_keys[1]: not rnd})

		self.quitopt = settings['quit']
		if self.quitopt['enable']:
			self.resp_keys.append(self.quitopt['button'])

		self.clock = core.Clock()
		self.current_trial = 0

		self.subject_id = 'test_subject'

		self.letters = list('bcdfghjklmnprstwxz')
		self.relations = ['>', '<']

		# wszystkie możliwe sekwencje
		# w wierszach warunki, w kolumnach wersje
		self.conditions = np.array( [[[1, 2, 2, 3, 3, 4],
			[3, 4, 2, 3, 1, 2]], [[2, 3, 1, 2, 3, 4],
			[2, 3, 3, 4, 1, 2]], [[1, 2, 3, 4, 2, 3],
			[3, 4, 1, 2, 2, 3]]] ) - 1
		# select rows and columns
		row_ind = self.settings['condition_rows']
		col_ind = self.settings['condition_columns']
		if len(col_ind) == 2:
			row_ind = [row_ind[0]] * 2 + [row_ind[1]] * 2
			col_ind = col_ind * 2
		self.conditions = self.conditions[row_ind, col_ind, :]
		self.conditions = self.conditions.reshape(
			len(self.settings['condition_rows']),
			len(self.settings['condition_columns']), 6)
		self.all_questions = [[[0,1], [1,2], [2,3]], [[0,2], [1,3]], [[0,3]]]

		self.create_stimuli()
		self.trials = self.create_trials(repetitions=self.settings['repetitions'])
		self.create_df()

	def get_time(self, stim):
		time = self.times[stim]
		if isinstance(time, list):
			return random.randint(*time)
		else:
			return time

	def check_quit(self, key=None):
	if self.quitopt['enable']:
		if key == None:
			key = event.getKeys()
		if key == None or len(key) == 0:
			return
		if isinstance(key[0], tuple):
			key = [k[0] for k in key]
		if isinstance(key, tuple):
			key, _ = key
		if self.quitopt['button'] in key:
			core.quit()

	def present_break(self):
		text = visual.TextStim(text=self.settings['tekst_przerwy'])
		k = False
		while not k:
			text.draw()
			self.window.flip()
			k = event.getKeys()

	def create_stimuli(self):
		args = {'units': 'deg', 'height': self.settings['sizes']['letter']}
		self.stim = {l: visual.TextStim(self.window, text=l.upper(), **args)
			for l in self.letters + self.relations + ['?']}

		# fixation cross/circle
		self.stim['fix'] = fix(self.window, height=self.settings['sizes']['fix_height'], 
			width=self.settings['sizes']['fix_width'], shape=self.settings['fix_shape'])
		if self.settings['fix_between_pairs']:
			self.stim.update(btw_pairs=self.stim['fix'])
		if self.settings['fix_during_wait']:
			self.stim.update(dur_wait=self.stim['fix'])

	def _show_pair(self, pair, times):
		# show_pair can return randomized times
		# elems = pair.split('')
		elems = list(pair)
		elems = [x.lower() for x in elems]
		assert len(elems) == len(times)

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
		times = self.show_pair(question)
		self.clock.reset()
		q_times = map(self.get_time, ['pre_question_mark', 'question_mark'])
		[self.show_element(el, tm) for el, tm in zip(['', '?'], q_times)]

		resp = event.waitKeys(maxWait=self.times['response'],
							  keyList=self.resp_keys,
					  		  timeStamped=self.clock)
		# return response
		if isinstance(resp, list):
			resp = resp[0]
		return (times, resp)

	def ask_questions(self, questions, feedback=None):
		output = list()
		for q in questions:
			# pre-question time?
			output.append(self.ask_question(q))
		if feedback is not None:
			print('nothing yet')
		return output

	def show_premises(self, model, sequence, relation):
		all_times = list()
		if isinstance(model, str):
			model = list(model)
		model = np.asarray(model)
		sequence = np.asarray(sequence)
		for i in [[0,1], [2,3], [4,5]]:
			pair = model[sequence[i]]
			pair = ' '.join([pair[0], relation, pair[1]])
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

	def show_trial(self, trial):
		# get model and relation
		model, sequence, relation = self.get_model(trial)

		# get questions
		questions = self.get_questions(model, relation, trial)
		self.filldf(trial, model, sequence, relation, questions)

		# show premises
		premise_times = self.show_premises(model, sequence, relation)

		# show questions
		time_and_resp = self.ask_questions(questions)
		self.save_responses(trial, time_and_resp)

	def reverse_relation(self, relation):
		if relation == '>':
			return '<'
		else:
			return '>'

	def _create_question(self, model, relation, distance, reverse, ifpos):
		if isinstance(model, str):
			model = list(model)
		model = np.asarray(model)

		q_pair = random.sample(self.all_questions[distance], 1)[0]
		if reverse:
			relation = self.reverse_relation(relation)
			q_pair.reverse()
		if not ifpos:
			q_pair.reverse()

		return ' '.join([model[q_pair[0]], relation, model[q_pair[1]]])

	def get_questions(self, model, relation, trial):
		chose_ind = self.trials['model'] == trial
		trial_df = self.trials.loc[chose_ind, :]
		questions = [self._create_question(model, relation,
			trial_df.loc[i, 'question_distance'],
			trial_df.loc[i, 'inverted_relation'],
			trial_df.loc[i, 'yesanswer']) for i in trial_df.index]
		return questions

	def get_model(self, trial):
		chose_ind = self.trials['model'] == trial
		trial_df = self.trials.loc[chose_ind, :]

		model = random.sample(self.letters, 4)
		relation = random.sample(self.relations, 1)[0]
		ii, jj = trial_df.iloc[0, [1,2]]
		sequence = self.conditions[ii, jj, :]
		return model, sequence, relation

	def filldf(self, trial, model, sequence, relation, questions):
		inds = np.where(self.trials['model'] == trial)[0]
		trial_df = self.trials.loc[inds, :]

		self.df.loc[inds, 'trial'] = trial
		self.df.loc[inds, 'model'] = ''.join(model)
		self.df.loc[inds, 'relation'] = relation
		cnd = int(np.all(sequence[-2:] == [1, 2]))
		self.df.loc[inds, 'condition'] = ['easy', 'difficult'][cnd]
		for i, ind in enumerate(inds):
			self.df.loc[ind, 'show_order'] = str(sequence)
			self.df.loc[ind, 'question'] = questions[i]
			self.df.loc[ind, 'question_distance'] = trial_df.loc[ind, 'question_distance']
			self.df.loc[ind, 'question_reversed'] = trial_df.loc[ind, 'inverted_relation']
			self.df.loc[ind, 'iftrue'] = trial_df.loc[ind, 'yesanswer']

	def save_responses(self, trial, time_and_resp):
		inds = np.where(self.trials['model'] == trial)[0]
		trial_df = self.trials.loc[inds, :]
		for i, elems in enumerate(time_and_resp):
			_, resp = elems
			if resp is not None:
				self.df.loc[inds[i], 'answer'] = int(self.resp_mapping[resp[0]])
				self.df.loc[inds[i], 'ifcorrect'] = int(self.resp_mapping[resp[0]] == \
					self.df.loc[inds[i], 'iftrue'])
				self.df.loc[inds[i], 'RT'] = resp[1]
			else:
				self.df.loc[inds[i], 'answer'] = np.nan
				self.df.loc[inds[i], 'ifcorrect'] = 0
				self.df.loc[inds[i], 'RT'] = np.nan

	def _create_combinations_matrix(self, repetitions=1):
		cnd_shp = self.conditions.shape
		mat = [[mrow, mcol, qtp, inv, yes] for mrow in range(cnd_shp[0])
			for mcol in range(cnd_shp[1]) for qtp in range(3)
			for inv in range(2) for yes in range(2)]
		return np.array(mat)

	def create_trials(self, repetitions=1):
		mat = self._create_combinations_matrix(repetitions=repetitions)
		nrow = mat.shape[0]
		df = pd.DataFrame(columns=['model', 'model_row', 'model_col',
			'question_distance', 'inverted_relation', 'yesanswer'],
			index=range(0, nrow))
		df = df.fillna(0)
		df_row = 0
		modelnum = 0
		while nrow > 0:
			modelnum += 1
			choose_model = random.randint(0, nrow-1)
			model = mat[choose_model, 0:2]
			same_model = np.where(np.all(mat[:, 0:2] == model, axis=1))[0]
			questions = mat[same_model, 2:]
			rm_row = list()
			question_order = random.sample(range(3), 3)
			for q in question_order:
				this_question = np.where(questions[:,0] == q)[0]
				pick_question = random.sample(list(this_question), 1)[0]
				df.iloc[df_row,:] = np.hstack([modelnum, model, questions[pick_question, :]])
				rm_row.append(same_model[pick_question])
				df_row += 1
			mat = np.delete(mat, rm_row, axis=0)
			nrow = mat.shape[0]
		return df

	def create_df(self):
		self.df = pd.DataFrame(columns=['trial', 'model',
			'show_order', 'relation', 'condition', 'question',
			'question_distance', 'question_reversed',
			'iftrue', 'answer', 'ifcorrect', 'RT'],
			index=range(0, self.trials.shape[0]))

	def save_data(self):
		fl = os.path.join('data', self.subject_id)
		self.df.to_excel(fl + '.xls')
		self.df.to_csv(fl + '.csv')


# stimuli
# -------
def fix(win, height=0.3, width=0.1, shape='circle', color=(0.5, 0.5, 0.5)):
	args = {'fillColor': color, 'lineColor': color,
		'interpolate': True, 'units': 'deg'}
	if shape == 'circle':
		fix_stim = visual.Circle(win, radius=height/2,
			edges=32, **args)
	else:
		h, w = (height/2, width/2)
		vert = np.array([[w, -h], [w, h], [-w, h], [-w, -h]])

		args.update(closeShape=True)
		fix_stim = [visual.ShapeStim(win, vertices=v, **args)
					for v in [vert, np.fliplr(vert)]]
	return fix_stim

# time
# ----
def get_frame_time(win, frames=25):
	frame_rate = win.getActualFrameRate(nIdentical = frames)
	if frame_rate is None:
		# try one more time
		frame_rate = win.getActualFrameRate(nIdentical = frames)
	return 1.0 / frame_rate

def s2frames(time_in_seconds, frame_time):
	assert isinstance(time_in_seconds, dict)
	time_in_frames = dict()
	toframes = lambda x: int(round(x / frame_time))
	for k, v in six.iteritems(time_in_seconds):
		if isinstance(v, list):
			time_in_frames[k] = map(toframes, v)
		else:
			time_in_frames[k] = toframes(v)
	return time_in_frames

# wypisz kolejne pary
# relation = random.sample(['<', '>'], 1)[0]
# for i in [[0,1], [2,3], [4,5]]:
# 	pair = sequence[current_cond[i]]
# 	print(pair[0], relation, pair[1])