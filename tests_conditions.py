# test LinOrd

from psychopy import visual
from exputils import LinOrdExperiment

window = visual.Window(monitor='testMonitor')

exp = LinOrdExperiment(window, 'settings.yaml')

# cond_mat = exp._create_combinations_matrix()
# print(cond_mat)

df = exp.create_trials(repetitions=1)
print(df.head(10))
df.to_excel('test_trials.xls')

question = exp._create_question('ABCD', '>', 2, 0, 1)
print(question) # 'A > D'
question = exp._create_question('ABCD', '>', 0, 1, 0)
print(question) # 'A < B' or 'B < C' etc.