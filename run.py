# test LinOrd

from psychopy import visual, monitors
from exputils import LinOrdExperiment

participantDistance = 60
monitor = monitors.Monitor('BENQ-XL2411', width=53., 
	distance=participantDistance)
monitor.setSizePix([1920, 1080])

window = visual.Window(monitor='testMonitor', units="deg",
	fullscr=False, size=[1200, 800])


exp = LinOrdExperiment(window, 'settings.yaml')
exp.get_subject_id()
exp.show_all_trials()
