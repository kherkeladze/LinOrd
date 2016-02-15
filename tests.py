# test LinOrd

from psychopy import visual
from exputils import LinOrdExperiment

window = visual.Window(monitor='testMonitor')

exp = LinOrdExperiment(window, 'settings.yaml')

exp.show_element('fix', 60)
exp.show_element('', 25)

exp.show_pair("h > w")

exp.show_element('fix', 80)
exp.show_premises('BCDF', [0,1,1,2,2,3], '<')

for i in range(4):
    exp.show_element('', 5)
    exp.show_element('fix', 5)

exp.show_trial(1)