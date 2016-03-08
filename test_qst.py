import os
from exputils import LinOrdExperiment
from psychopy import visual

os.chdir(r'D:\proj\LinOrd')
win = visual.Window(monitor='testMonitor')
exp = LinOrdExperiment(win, 'settings.yaml')
df = exp.create_trials()
df.head(10)

for i in range(3):
	qst = exp.get_questions('ABCD', '<', i)
	for q in qst:
		print(q)
	exp.df.head(10)
	exp.trials.head(10)
