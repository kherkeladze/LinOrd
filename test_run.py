# -*- encoding: utf-8 -*-
# test LinOrd
from __future__ import division, print_function
import os
import random
from psychopy import core, visual, monitors, event
import exputils
from exputils import LinOrdExperiment, Instructions

print(os.getcwd())
print(exputils.__file__)

participantDistance = 60
file_path = os.path.join(*(__file__.split('\\')[:-1]))
file_path = file_path.replace(':', ':\\')

# check correct monitor type
monitorList = monitors.getAllMonitors()
if 'BENQ-XL2411' in monitorList:
    monitor = monitors.Monitor('BENQ-XL2411', width=53., 
        distance=participantDistance)
    monitor.setSizePix([1920, 1080])
else:
    monitor = 'testMonitor'

# create temporary window
window = visual.Window(monitor=monitor, units="deg", fullscr=True)
exp = LinOrdExperiment(window, os.path.join(file_path, 'settings.yaml'))

#create fullscreen window

waitText = visual.TextStim(window, text=u'Proszę czekać...', height=2)
waitText.draw(); window.flip()
# hide mouse
window.setMouseVisible(False)

print(exp.resp_keys)
print(exp.resp_mapping)

# instructions
os.chdir(file_path)
instr_dir = os.path.join(file_path, 'instr')
instr = os.listdir(instr_dir)
resp_slides = ['porzadki6_F-TAK.png', 'porzadki6_J-TAK.png']
if exp.resp_mapping['f']:
    instr = [resp_slides[0]]
else:
    instr = [resp_slides[1]]
instr = [os.path.join('instr', i) for i in instr]

instr = Instructions(window, instr)
instr.present()

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

exp.show_all_trials()
