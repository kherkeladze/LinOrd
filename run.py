# -*- encoding: utf-8 -*-
# test LinOrd
from __future__ import division
import os
import random
from psychopy import core, visual, monitors, event
from exputils import LinOrdExperiment, Instructions


# how far is participant from the screen?
scr_dist = 60

def get_screen(scr_dist=scr_dist):
    # check correct monitor type
    monitorList = monitors.getAllMonitors()
    if 'BENQ-XL2411' in monitorList:
        monitor = monitors.Monitor('BENQ-XL2411', width=53.,
            distance=scr_dist)
        monitor.setSizePix([1920, 1080])
    else:
        monitor = 'testMonitor'
    return monitor


def go_to_file_dir(pth):
    file_path = os.path.join(*(pth.split('\\')[:-1]))
    file_path = file_path.replace(':', ':\\')
    os.chdir(file_path)


def run(window=None, subject_id=None, true_key=None,
        scr_dist=scr_dist):

    # set path to current file location
    go_to_file_dir(__file__)

    # create window
    if window is None:
        monitor = get_screen(scr_dist=scr_dist)
        window = visual.Window(monitor=monitor, units="deg", fullscr=True)

    exp = LinOrdExperiment(window, os.path.join(file_path, 'settings.yaml'))
    exp.set_subject_id(subject_id)
    exp.set_resp(true_key=true_key)

    waitText = visual.TextStim(window, text=u'Proszę czekać...', height=2)
    # exp.set_window(window)
    waitText.draw(); window.flip()

    # hide mouse
    window.mouseVisible = False

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
    for i in range(1, exp.settings['training_trials'] + 1):
        exp.show_trial(i, feedback=True)
        exp.save_data()
        if i > 1 and exp.df.loc[i, 'ifcorrect'] == 0:
            exp.show_keymap()

    exp.subject['id'] = orig_subj_id
    exp.create_trials(repetitions=exp.settings['repetitions'])

    instr.present(stop=5)

    exp.show_all_trials()
    instr.present()
    return exp


if __name__ == '__main__':
    run()