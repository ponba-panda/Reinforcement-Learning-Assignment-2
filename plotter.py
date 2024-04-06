import numpy as np
import matplotlib.pyplot as plt

soft_return = np.loadtxt('CartPole-v1_max.csv', delimiter=',')

mean_data = np.mean(soft_return, axis=0)
variance_data = np.var(soft_return, axis=0)
shade_data_1ow = mean_data - np.sqrt(variance_data)
shade_data_high = mean_data + np.sqrt(variance_data)

plt.plot(mean_data, label='max')
plt.fill_between(range(soft_return.shape[1]), shade_data_1ow, shade_data_high, alpha=0.3)
# plt.plot(best_return*np.ones(total_episodes,))

soft_return = np.loadtxt('CartPole-v1_average.csv', delimiter=',')

mean_data = np.mean(soft_return, axis=0)
variance_data = np.var(soft_return, axis=0)
shade_data_1ow = mean_data - np.sqrt(variance_data)
shade_data_high = mean_data + np.sqrt(variance_data)

plt.plot(mean_data, label='average')
plt.fill_between(range(soft_return.shape[1]), shade_data_1ow, shade_data_high, alpha=0.3)

plt.plot()
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title(f'Dueling DQN for CartPole-v1 - Returns per Episode')
plt.legend()
plt.savefig('CartPole-v1_duelingDQN.pdf')
plt.show()

soft_return = np.loadtxt('Acrobot-v1_max.csv', delimiter=',')

mean_data = np.mean(soft_return, axis=0)
variance_data = np.var(soft_return, axis=0)
shade_data_1ow = mean_data - np.sqrt(variance_data)
shade_data_high = mean_data + np.sqrt(variance_data)

plt.plot(mean_data, label='max')
plt.fill_between(range(soft_return.shape[1]), shade_data_1ow, shade_data_high, alpha=0.3)
# plt.plot(best_return*np.ones(total_episodes,))

soft_return = np.loadtxt('Acrobot-v1_average.csv', delimiter=',')

mean_data = np.mean(soft_return, axis=0)
variance_data = np.var(soft_return, axis=0)
shade_data_1ow = mean_data - np.sqrt(variance_data)
shade_data_high = mean_data + np.sqrt(variance_data)

plt.plot(mean_data,label='average')
plt.fill_between(range(soft_return.shape[1]), shade_data_1ow, shade_data_high, alpha=0.3)

plt.plot()
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title(f'Dueling DQN for Acrobot-v1 - Returns per Episode')
plt.legend()
plt.savefig('Acrobot-v1_duelingDQN.pdf')
plt.show()
