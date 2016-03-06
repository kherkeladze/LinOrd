import os
from exputils import LinOrdExperiment
from psychopy import visual

os.chdir(r'D:\proj\LinOrd')
win = visual.Window('testMonitor')
exp = LinOrdExperiment(win, 'settings.yaml')
df = exp.create_trials()
df.head(10)

qst = exp.get_questions('ABCD', '<', 1)
for q in qst:
	print(q)
exp.trials.head(10)

qst = exp.get_questions('ABCD', '<', 2)
for q in qst:
	print(q)
exp.trials.head(10)