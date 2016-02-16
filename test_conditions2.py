from psychopy import visual
from exputils import LinOrdExperiment

window = visual.Window(monitor='testMonitor')

exp = LinOrdExperiment(window, 'settings.yaml')

model, sequence, relation = exp.get_model(1)
print(model)
print(sequence)
print(relation)

# change colors to isoluminant
exp.stim['<'].setColor([0,135,1], colorSpace='dkl')
exp.stim['>'].setColor([0,315,1], colorSpace='dkl')

exp.show_keymap()
exp.show_trial(1)
exp.present_break()
exp.show_keymap()
exp.show_trial(2)
exp.save_data()