from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym_super_mario_bros import smb_random_stages_env, smb_env
import random
import numpy as np


from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
import os


EPISODES = 3000

batch_size = 16

intento = input("número del intento: ")

guardar = './save/red_mario_entrenamiento'+intento+'.h5'


class DQNAgent:

	def crear_red(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.03
		self.model = self._build_model()

	def _build_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		states, targets_f = [], []
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			states.append(state[0])
			targets_f.append(target_f[0])
		history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
		loss = history.history['loss'][0]
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
		return loss

	def load(self, name):
		self.model.load_weights(name)
		print("load! ")

	def save(self, name):
		global intento
		model_json = self.model.to_json()
		print("abriendo")
		with open("model_entrenamiento"+intento+".json", "w") as json_file:
			print("escribiendo...")
			json_file.write(model_json)
		print("listo!")
		self.model.save_weights(name)


env1 = gym_super_mario_bros.make('SuperMarioBros-3-2-v0')
env = BinarySpaceToDiscreteSpaceEnv(env1, COMPLEX_MOVEMENT)

state_size = 180
action_size = env.action_space.n

agent = DQNAgent()
agent.crear_red(state_size, action_size)
done = False

max_reward = 0
"""for i in range(4):
env1 = gym_super_mario_bros.make('SuperMarioBros-4-'+str(i+1)+'-v0')
env = BinarySpaceToDiscreteSpaceEnv(env1, COMPLEX_MOVEMENT)
EPISODES = 100
done = False """
ETAPAS = ['1-1', '2-1']

for i in ETAPAS:
    env1 = gym_super_mario_bros.make('SuperMarioBros-'+i+'-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env1, COMPLEX_MOVEMENT)
    done = False
    eps = EPISODES//2
    for e in range(eps):
        state = env.reset()
        _, _, _, info = env.step(0)
        total_reward = 0
        state = np.reshape(info["enemy"], [1, state_size])
        checkpoint = info["x_pos"] + 100
        contador = 1
        for time in range(5000):
            #env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            reward = info["reward"]
            if info["x_pos"] >= checkpoint:
                reward += 3000 + 500*contador
                checkpoint += 100
                contador += 1
            total_reward += reward
            next_state = np.reshape(info["enemy"], [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                if total_reward > max_reward: max_reward = total_reward
                print("episode {}/{}, score: {}, e:{:.2} max reward: {}".format(e, EPISODES, total_reward, agent.epsilon, max_reward))
                break

            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                if time % 10 == 0:
                    print("episode {}/{}, time: {}, loss: {:.4f}".format(e, EPISODES, time, loss))
        if e != 0 and e%500 == 0: agent.save(guardar)
        if e == 0: max_reward = total_reward
        print("Total Reward: {}, Max reward: {}".format(total_reward, max_reward))
    env.close()
agent.save(guardar)