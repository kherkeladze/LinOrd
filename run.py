# -*- encoding: utf-8 -*-
# test LinOrd
from __future__ import division
import os
import random

import numpy as np
from psychopy import core, visual, monitors, event
from exputils import LinOrdExperiment, Instructions


# how far is participant from the screen?
scr_dist = 60

def get_screen(scr_dist=scr_dist):
    return 'testMonitor'


def go_to_file_dir(pth):
    file_path = os.path.join(*(pth.split('\\')[:-1]))
    file_path = file_path.replace(':', ':\\')
    os.chdir(file_path)
    return file_path


def run(window=None, subject_id=None, true_key=None,
        scr_dist=scr_dist):

    # set path to current file location
    file_path = go_to_file_dir(__file__)

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
        del instr[5]
    else:
        del instr[4]
    instr = [os.path.join(instr_dir, f) for f in instr]

    instr = Instructions(window, instr)
    instr.present(stop=6)

    orig_subj_id = exp.subject['id']
    exp.subject['id'] += '_training'
    exp.trials = exp.trials.query('model_row == 0').reset_index(drop=True)
    # reset model id
    nrows = exp.trials.shape[0]
    exp.trials.loc[:, 'model'] = np.tile(np.arange(1, nrows/3 + 1, dtype='int'),
                                [3, 1]).T.ravel()

    # training
    for i in range(1, exp.settings['training_trials'] + 1):
        exp.show_trial(i, feedback=True)
        exp.save_data()
        if i > 1 and exp.df.loc[i, 'ifcorrect'] == 0:
            exp.show_keymap()

    exp.subject['id'] = orig_subj_id
    exp.create_trials(repetitions=exp.settings['repetitions'])
    
    # ask if everything is clear
    args = {'units': 'deg', 'height': exp.settings['sizes']['key_info']}
    text = u'Jeżeli masz jakieś pytania/wątpliwości dotyczące zadania, ' + \
        u'możesz zapytać się eksperymentatora.\nJeżeli nie masz żadnych' + \
        u'pytań, możesz przejść dalej naciskając spację'
    text = visual.TextStim(exp.window, text=text, **args)
    text.wrapWidth = 20
    text.draw(); exp.window.flip()
    k = event.waitKeys(keyList=['space'])

    instr.present(stop=7)
    exp.show_all_trials()
    return exp


if __name__ == '__main__':
    run()
