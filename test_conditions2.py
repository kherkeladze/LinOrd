from psychopy import visual
from exputils import LinOrdExperiment

window = visual.Window(monitor='testMonitor')

exp = LinOrdExperiment(window, 'settings.yaml')

model, sequence, relation = exp.get_model(1)
print(model)
print(sequence)
print(relation)

exp.show_trial(1)
exp.show_trial(2)
exp.save_data()