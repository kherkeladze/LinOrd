from __future__ import print_function
import os
from exputils import LinOrdExperiment
from psychopy import visual

os.chdir(r'E:\PROJ\EXP\LinOrd')
win = visual.Window(monitor='testMonitor')
exp = LinOrdExperiment(win, 'settings.yaml')
df = exp.create_trials()
print(df.head(10))

qst = exp.get_questions('ABCD', '<', 1)
for q in qst:
	print(q)
exp.filldf(1, 'ABCD', [[0,1],[1,2],[2,3]], '<', qst)
print(exp.df.head(10))

qst = exp.get_questions('ABCD', '<', 2)
for q in qst:
	print(q)
exp.filldf(1, 'ABCD', [[0,1],[1,2],[2,3]], '<', qst)
print(exp.df.head(10))
print(exp.trials.head(10))