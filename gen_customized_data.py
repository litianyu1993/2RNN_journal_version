import numpy as np

'''
Generating customized data with single trajectory
'''
traj_length = 100
train = np.ones(traj_length) + np.random.normal(0, 0.1, (traj_length))
test = np.ones(traj_length) + np.random.normal(0, 0.1, (traj_length))
np.savetxt('./Data/customized/single_traj/train_data.csv', train, delimiter=',')
np.savetxt('./Data/customized/single_traj/test_data.csv', test, delimiter=',')

'''
Generating customized data with multiple trajectory
'''
