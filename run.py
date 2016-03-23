# -*- encoding: utf-8 -*-
# test LinOrd
from __future__ import division
import os
import random
from psychopy import core, visual, monitors, event
from exputils import LinOrdExperiment, Instructions

participantDistance = 60
file_path = os.path.join(*(__file__.split('\\')[:-1]))
file_path = file_path.replace(':', ':\\')
print file_path

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

exp = LinOrdExperiment(window, os.path.join(file_path, 'settings.yaml'))
exp.get_subject_id()

#create fullscreen window
window = visual.Window(monitor=monitor, units="deg", fullscr=True)
waitText = visual.TextStim(window, text=u'Proszę czekać...', height=2)
exp.set_window(window)
waitText.draw(); window.flip()
# hide mouse
window.setMouseVisible(False)

# instructions
instr_dir = os.path.join(os.getcwd(), 'instr')
instr = os.listdir(instr_dir)
if exp.isolum['>'] == 135:
    todel = [1,3,5,8]
    todel.reverse()
    for d in todel:
        del instr[d]
else:
    todel = [0,2,4,7]
    todel.reverse()
    for d in todel:
        del instr[d]

if exp.resp_mapping['f']:
    del instr[6]
else:
    del instr[5]
instr = [os.path.join('instr', i) for i in instr]

# add examples to instructions
def example1():
    exp.show_pair('D > B')
    core.wait(0.35)

def example2():
    exp.show_premises('BGPZ', [0,1,1,2,2,3],
        '>', with_wait=False)
    core.wait(0.35)
    waitText.setText('poprawny model to:\nB > G > P > Z')
    waitText.draw()
    window.flip()
    event.waitKeys()

def example3():
    exp.show_trial(random.randint(4, exp.df.shape[0]/3-1))

instr.insert(1, example1)
instr.insert(4, example2)
instr.insert(8, example3)
instr = Instructions(window, instr)
instr.present(stop=10)

orig_subj_id = exp.subject['id']
exp.subject['id'] += '_training'
# training
for i in range(1, 8):
    exp.show_trial(i, feedback=True)
    exp.save_data()
    if i > 1 and exp.df.loc[i, 'ifcorrect'] == 0:
        exp.show_keymap()

exp.subject['id'] = orig_subj_id
exp.create_trials(repetitions=exp.settings['repetitions'])

instr.present(stop=11)

exp.show_all_trials()
instr.present()
