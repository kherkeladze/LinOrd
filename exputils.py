# -*- encoding: utf-8 -*-
from __future__ import division, print_function

import os
import six
import yaml
import types
import random
import numpy as np
import pandas as pd

from psychopy import visual, event, core, gui

# TODOs:
# - [ ] resolve pre-relation time and pre-question time?
# - [ ] configurable feedback time
# - [ ] test markers


class LinOrdExperiment(object):
	def __init__(self, window, config_file):
		# super class should do this:
		self.window = window
		self.frame_time = get_frame_time(window)

		# load settings
		file_name = os.path.join(os.getcwd(), config_file)
		with open(file_name, 'r') as f:
			settings = yaml.load(f)

		self.sequential = settings['present_sequential']

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

		self.subject =  dict()
		self.subject['id'] = 'test_subject'

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

		# port stuff
		self.send_triggers = self.settings['send_triggers']
		self.port_adress = self.settings['port_adress']
		self.triggers = self.settings['triggers']
		self.set_up_ports()

		self.isolum = dict()
		self.create_stimuli()
		self.trials = self.create_trials(repetitions=self.settings['repetitions'])
		self.create_df()
		self.num_trials = int(np.max(self.trials['model']))

	def set_window(self, window):
		self.window = window
		for s in self.stim.values():
			s.win = window

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

	# DISPLAY
	# -------
	def show_all_trials(self):
		trials_without_break = 0
		self.show_keymap()
		for t in range(1, self.num_trials+1):
			self.show_trial(t)
			self.save_data()
			trials_without_break += 1
			if trials_without_break >= self.settings['przerwa_co_ile_modeli']:
				trials_without_break = 0
				self.present_break()
				self.show_keymap()

	def show_trial(self, trial, feedback=False):
		# get model and relation
		model, sequence, relation = self.get_model(trial)

		# get questions
		questions = self.get_questions(model, relation, trial)
		self.filldf(trial, model, sequence, relation, questions)

		# show premises
		if self.send_triggers:
			for el in ['letter', 'relation']:
				self.triggers[el] = self.settings['triggers'][el]
		premise_times = self.show_premises(model, sequence, relation)

		# change triggers for questions
		if self.send_triggers:
			add = self.settings['triggers']['question_add']
			for el in ['letter', 'relation']:
				self.triggers[el] = self.settings['triggers'][el] + add

		# show questions
		for q_num, q in enumerate(questions):
			time_and_resp = self.ask_question(q)
			if feedback:
				# calculate dataframe row
				row = (trial - 1) * 3 + q_num
				# get and check response
				response = time_and_resp[1][0]
				if response is not None:
					response = self.df.loc[row, 'iftrue'] == self.\
						resp_mapping[response]
				else:
					response = False
				# choose relevant circle and show
				circ = 'feedback_' + ['in',''][int(response)] + 'correct'
				self.show_element(circ, 25)	
				core.wait(0.25)
				self.window.flip()
			self.save_responses(trial, q_num, time_and_resp)
		finish_time = self.get_time('after_last_question')
		self.show_element('btw_pairs', finish_time)

	def show_premises(self, model, sequence, relation, with_wait=True):
		all_times = list()
		if isinstance(model, str):
			model = list(model)
		model = np.asarray(model)
		sequence = np.asarray(sequence)
		for i in [[0,1], [2,3], [4,5]]:
			pair = model[sequence[i]]
			pair = ' '.join([pair[0], relation, pair[1]])

			# pre-stim time
			time1 = self.get_time('pre_pair')
			self.show_element('btw_pairs', time1)

			# show pair
			times = self.show_pair(pair)
			times = [time1] + times
			if i[0] is not 4:
				next_time1 = self.get_time('after_pair')
				next_time2 = 0
				self.show_element('btw_pairs', next_time1)
			elif with_wait:
				next_time1 = self.get_time('after_last_pair')
				next_time2 = self.get_time('fix_highlights')
				next_time1 -= next_time2
				self.show_element('dur_wait', next_time1)
				self.show_element('dur_wait_change', next_time2)
				self.check_quit()
			else:
				next_time1, next_time2 = 0, 0
			# add to times and append to all_times
			times += [next_time1, next_time2]
		all_times.append(times)
		return np.array(all_times)

	def ask_question(self, question):
		# show relation
		if not self.sequential:
			question += ' ?'

		# pre-stim time
		time1 = self.get_time('pre_pair')
		self.show_element('btw_pairs', time1)

		# show pair
		times = self.show_pair(question)
		self.clock.reset()
		times = [time1] + times
		if self.sequential:
			q_times = map(self.get_time, ['pre_question_mark', 'question_mark'])
			[self.show_element(el, tm) for el, tm in zip(['', '?'], q_times)]

		resp = event.waitKeys(maxWait=self.times['response'],
							  keyList=self.resp_keys,
					  		  timeStamped=self.clock)
		# return response
		self.check_quit(key=resp)
		if isinstance(resp, list):
			resp = resp[0]
		return (times, resp)

	def show_pair(self, pair):
		if self.sequential:
			events = ['element', 'before_relation', 'relation',
				'after_relation', 'element']
			times = map(self.get_time, events)
		else:
			times = [self.get_time('element')]
		self._show_pair(pair, times)
		return times

	def _show_pair(self, pair, times):
		# show_pair can return randomized times
		# elems = pair.split('')
		elems = list(pair)
		if self.sequential:
			elems = [x.lower() for x in elems]
			assert len(elems) == len(times)
			[self.show_element(el, tm) for el, tm in zip(elems, times)]
		else:
			# set position of left and right letter
			elems = [x.lower() for x in elems if x != ' ']
			itr = [0,2,3] if len(elems)==4 else [0, 2]
			for i in itr:
				self.stim[elems[i]].pos = [self.settings['elem_x_pos'][i], 0.]
			self.show_element(elems, times[0])
		self.check_quit()

	def present_break(self):
		text = self.settings['tekst_przerwy']
		text = text.replace('\\n', '\n')
		text = visual.TextStim(self.window, text=text)
		k = False
		while not k:
			text.draw()
			self.window.flip()
			k = event.getKeys()
		self.check_quit(key=k)

	def show_keymap(self):
		args = {'units': 'deg', 'height': self.settings['sizes']['key_info']}
		show_map = {k: bool_to_pl(v)
			for k, v in six.iteritems(self.resp_mapping)}
		text = u'Odpowiadasz klawiszami:\nf: {}\nj: {}'.format(
			show_map['f'], show_map['j'])
		stim = visual.TextStim(self.window, text=text, **args)
		stim.draw()
		self.window.flip()
		k = event.waitKeys()
		self.check_quit(key=k)

	def show_element(self, elem, time):
		elem_show = True
		is_list = isinstance(elem, list)
		if not is_list and elem not in self.stim:
			elem_show = False
		# draw element
		if elem_show:
			if not is_list:
				elem = [elem]
			self.set_trigger(elem[0])
		for f in range(time):
			if elem_show:
				if f == 2:
					self.set_trigger(0)
				for el in elem:
					self.stim[el].draw()
			self.window.flip()

	# other
	# -----
	def create_stimuli(self):
		args = {'units': 'deg', 'height': self.settings['sizes']['letter']}
		self.stim = {l: visual.TextStim(self.window, text=l.upper(), **args)
			for l in self.letters + self.relations + ['?']}

		# take care of isoluminant relations
		if self.settings['isoluminant_relations']:
			deg = random.sample(self.settings['iso_degs'], 2)
			for r, d in zip(self.relations, deg):
				self.stim[r].setColor([0,d,1], colorSpace='dkl')
				self.isolum[r] = d

		# fixation cross/circle
		self.stim['fix'] = fix(self.window, height=self.settings['sizes']['fix_height'],
			width=self.settings['sizes']['fix_width'], shape=self.settings['fix_shape'])
		if self.settings['fix_between_pairs']:
			self.stim.update(btw_pairs=self.stim['fix'])
			self.triggers['btw_pairs'] = self.triggers['fixation'][0]
		if self.settings['fix_during_wait']:
			self.triggers['dur_wait'] = self.triggers['fixation'][0]
			self.triggers['dur_wait_change'] = self.triggers['fixation'][1]
			self.stim.update(dur_wait=self.stim['fix'])

			change_color = np.array(self.settings['fix_change_color'])
			change_color = (change_color / 255. - 0.5) * 2.
			change_fix= fix(self.window, height=self.settings['sizes']['fix_height'],
				width=self.settings['sizes']['fix_width'], shape=self.settings['fix_shape'],
				color=change_color)
			self.stim.update(dur_wait_change=change_fix)

		# feedback circles
		feedback_colors = (np.array([[0,147,68], [190, 30, 45]],
			dtype='float') / 255 - 0.5) * 2
		self.stim['feedback_correct'] = fix(self.window, height=self.settings[
			'feedback_circle_radius'], color=feedback_colors[0,:])
		self.stim['feedback_incorrect'] = fix(self.window, height=self.\
			settings['feedback_circle_radius'], color=feedback_colors[1,:])

	def reverse_relation(self, relation):
		if relation == '>':
			return '<'
		else:
			return '>'

	# create trials, save responses
	# -----------------------------
	def get_model(self, trial):
		chose_ind = self.trials['model'] == trial
		trial_df = self.trials.loc[chose_ind, :]

		model = random.sample(self.letters, 4)
		relation = random.sample(self.relations, 1)[0]
		ii, jj = trial_df.iloc[0, [1,2]]
		sequence = self.conditions[ii, jj, :]
		return model, sequence, relation

	def get_questions(self, model, relation, trial):
		chose_ind = self.trials['model'] == trial
		trial_df = self.trials.loc[chose_ind, :]
		questions = [self._create_question(model, relation,
			trial_df.loc[i, 'question_distance'],
			trial_df.loc[i, 'inverted_relation'],
			trial_df.loc[i, 'yesanswer']) for i in trial_df.index]
		return questions

	def _create_question(self, model, relation, distance, reverse, ifpos):
		if isinstance(model, str):
			model = list(model)
		model = np.asarray(model)

		q_pair = list(random.sample(self.all_questions[distance], 1)[0])

		if reverse:
			relation = self.reverse_relation(relation)
			q_pair.reverse()
		if not ifpos:
			q_pair.reverse()

		question = ' '.join([model[q_pair[0]], relation, model[q_pair[1]]])
		return question

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

	def _create_combinations_matrix(self, repetitions=1):
		cnd_shp = self.conditions.shape
		mat = [[mrow, mcol, qtp, inv, yes] for mrow in range(cnd_shp[0])
			for mcol in range(cnd_shp[1]) for qtp in range(3)
			for inv in range(2) for yes in range(2)] * repetitions
		return np.array(mat)

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

	def save_responses(self, trial, question_num, time_and_resp):
		ind = np.where(self.trials['model'] == trial)[0][question_num]
		_, resp = time_and_resp
		if resp is not None:
			self.df.loc[ind, 'answer'] = int(self.resp_mapping[resp[0]])
			self.df.loc[ind, 'ifcorrect'] = int(self.resp_mapping[resp[0]] == \
				self.df.loc[ind, 'iftrue'])
			self.df.loc[ind, 'RT'] = resp[1]
		else:
			self.df.loc[ind, 'answer'] = np.nan
			self.df.loc[ind, 'ifcorrect'] = 0
			self.df.loc[ind, 'RT'] = np.nan

	def _create_combinations_matrix(self, repetitions=1):
		cnd_shp = self.conditions.shape
		mat = [[mrow, mcol, qtp, inv, yes] for mrow in range(cnd_shp[0])
			for mcol in range(cnd_shp[1]) for qtp in range(3)
			for inv in range(2) for yes in range(2)] * repetitions
		return np.array(mat)

	def create_trials(self, repetitions=1):
		mat = self._create_combinations_matrix(repetitions=repetitions)
		np.random.shuffle(mat)
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
		fl = os.path.join('data', self.subject['id'])
		self.df.to_excel(fl + '.xls')
		self.df.to_csv(fl + '.csv')

	def get_subject_id(self):
		myDlg = gui.Dlg(title="Subject Info", size = (800,600))
		myDlg.addText('Informacje o osobie badanej')
		myDlg.addField('ID:')
		myDlg.addField('wiek:', 30)
		myDlg.addField(u'płeć:', choices=[u'kobieta', u'mężczyzna'])
		myDlg.show()  # show dialog and wait for OK or Cancel

		if myDlg.OK:  # Ok was pressed
			self.subject['id'] = myDlg.data[0]
			self.subject['age'] = myDlg.data[1]
			self.subject['sex'] = myDlg.data[2]
		else:
			core.quit()

	# triggers
	# --------
	def set_up_ports(self):
		if self.send_triggers:
			try:
				from ctypes import windll
				windll.inpout32.Out32(self.port_adress, 111)
				core.wait(0.1)
				windll.inpout32.Out32(self.port_adress, 0)
				self.inpout32 = windll.inpout32
			except:
				warnings.warn('Could not send test trigger. :(')
				self.send_triggers = False

	def send_trigger(self, code):
		self.inpout32.Out32(self.port_address, code)

	def set_trigger(self, event):
		if self.send_triggers:
			if isinstance(event, int):
				self.window.callOnFlip(self.send_trigger, event)
			else:
				if event in self.letters:
					trig = self.triggers['letter']
					self.window.callOnFlip(self.send_trigger, trig)
				elif event in self.relations:
					trig = self.triggers['relation']
					self.window.callOnFlip(self.send_trigger, trig)
				elif event in self.triggers:
					trig = self.triggers[event]
					self.window.callOnFlip(self.send_trigger, trig)
				elif event == '?':
					trig = self.triggers[question_mark]
					self.window.callOnFlip(self.send_trigger, trig)


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

def bool_to_pl(b):
	assert isinstance(b, bool)
	return ['NIE', 'TAK'][int(b)]


class Instructions:
	def __init__(self, win, instrfiles):
		self.win = win
		self.nextpage   = 0
		self.navigation = {'left': 'prev', 'right': 'next',
			'space': 'next'}

		# get instructions from file:
		self.imagefiles = instrfiles
		self.images = []
		self.generate_images()
		self.stop_at_page = len(self.images)

	def generate_images(self):
		self.images = []
		for imfl in self.imagefiles:
			if not isinstance(imfl, types.FunctionType):
				self.images.append(visual.ImageStim(self.win,
					image=imfl, size=[1169, 826], units='pix',
					interpolate=True))
			else:
				self.images.append(imfl)

	def present(self, start=None, stop=None):
		if not isinstance(start, int):
			start = self.nextpage
		if not isinstance(stop, int):
			stop = len(self.images)

		# show pages:
		self.nextpage = start
		while self.nextpage < stop:
			# create page elements
			action = self.show_page()

			# go next/prev according to the response
			if action == 'next':
				self.nextpage += 1
			else:
				self.nextpage = max(0, self.nextpage - 1)

	def show_page(self, page_num=None):
		if not isinstance(page_num, int):
			page_num = self.nextpage

		img = self.images[page_num]
		if not isinstance(img, types.FunctionType):
			img.draw()
			self.win.flip()

			# wait for response
			k = event.waitKeys(keyList=self.navigation.keys())[0]
			return self.navigation[k]
		else:
			img()
			return 'next'
