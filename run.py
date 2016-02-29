# -*- encoding: utf-8 -*-
# test LinOrd
import os
from psychopy import core, visual, monitors
from exputils import LinOrdExperiment, Instructions

participantDistance = 60

# check correct monitor type
monitorList = monitors.getAllMonitors()
if 'BENQ-XL2411' in monitorList:
    monitor = monitors.Monitor('BENQ-XL2411', width=53., 
        distance=participantDistance)
    monitor.setSizePix([1920, 1080])
else:
    monitor = 'testMonitor'

# create temporary window
window = visual.Window(monitor=monitor, units="deg",
	fullscr=False, size=[1200,800])

exp = LinOrdExperiment(window, 'settings.yaml')
exp.get_subject_id()

#create fullscreen window
window = visual.Window(monitor=monitor, units="deg", fullscr=True)
waitText = visual.TextStim(window, text=u'Proszę czekać...', height=2)
exp.set_window(window)
waitText.draw(); window.flip()

# instructions
instr_dir = os.path.join(os.getcwd(), 'instr')
instr = os.listdir(instr_dir)
if exp.isolum['>'] == 135:
    del instr[1]
else:
    del instr[0]
if exp.resp_mapping['f']:
    del instr[4]
else:
    del instr[3]
instr = [os.path.join('instr', i) for i in instr]

# add examples to instructions
def example():
    exp.show_premises('BGPZ', [0,1,1,2,2,3],
        '>', with_wait=False)
    core.wait(0.5)

instr.insert(1, example)
instr = Instructions(window, instr)
instr.present(stop=6)

# training
for i in range(1, 16):
    exp.show_trial(i, feedback=True)
    if i > 1 and exp.df.loc[i, 'ifcorrect'] == 0:
        exp.show_keymap()
exp.create_trials(repetitions=exp.settings['repetitions'])

instr.present(stop=7)

exp.show_all_trials()
instr.present()
