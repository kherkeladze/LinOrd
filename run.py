# -*- encoding: utf-8 -*-
# test LinOrd
from __future__ import division
import os
import random
from psychopy import core, visual, monitors, event
from exputils import LinOrdExperiment, Instructions

participantDistance = 60

# set path
file_path = os.path.join(*(__file__.split('\\')[:-1]))
file_path = file_path.replace(':', ':\\')
os.chdir(file_path)

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

# at least now:
exp.set_resp(true_key='f')

# instructions
instr_dir = os.path.join(os.getcwd(), 'instr')
instr = os.listdir(instr_dir)

if exp.resp_mapping['f']:
    del instr[2]
else:
    del instr[1]
instr = [os.path.join(instr_dir, f) for f in instr]

instr = Instructions(window, instr)
instr.present(stop=4)

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

instr.present(stop=5)

exp.show_all_trials()
instr.present()
