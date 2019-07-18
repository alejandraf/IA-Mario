from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym_super_mario_bros import smb_random_stages_env, smb_env
import random
import numpy as np
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
import os

#n = input("Modelo (intento): ")
n = '7'
cargar_pesos = './save/red_mario_entrenamiento'+n+'.h5'
modelo = 'model_entrenamiento'+n+'.json'

env1 = gym_super_mario_bros.make('SuperMarioBros-5-1-v0')
env = BinarySpaceToDiscreteSpaceEnv(env1, COMPLEX_MOVEMENT)

state_size = 180
action_size = env.action_space.n
learning_rate = 0.03
epsilon = 0.01

json_file = open(modelo, "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights(cargar_pesos)

model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

done = False
batch_size = 16
"""
state = env.reset()
_, _, _, info = env.step(0)
state = np.reshape(info["enemy"], [1, state_size])
"""
for i in range(8):
    for j in range(4):
        env1 = gym_super_mario_bros.make('SuperMarioBros-'+str(i+1)+'-'+str(j+1)+'-v0')
        env = BinarySpaceToDiscreteSpaceEnv(env1, COMPLEX_MOVEMENT)
        state = env.reset()
        _, _, _, info = env.step(0)
        total_reward = 0
        reward_checkpoint = 3000
        state = np.reshape(info["enemy"], [1, state_size])
        #state = np.append(state,info["time"])
        #state = np.reshape(state, [1, state_size+1])
        checkpoint = info["x_pos"] + 50
        done = False
        x0 = info["x_pos"]
        y0 = info["y_pos"]
        t0 = info["time"]
        t = 0
        for k in range(10):
            quieto = 0
            total_reward = 0
            while not done:
                env.render()
                if np.random.rand() <= epsilon:
                    action = random.randrange(action_size)
                else:
                    act_values = model.predict(state)
                    action = np.argmax(act_values[0])

                next_state, reward, done, info = env.step(action)
                x = info["x_pos"]
                y = info["y_pos"]
                if x == x0:
                    quieto += 1
                    if quieto > 1000: break
                reward = info["reward"]
                if info["x_pos"] >= checkpoint:
                	reward += reward_checkpoint
                	checkpoint += 50
                	reward_checkpoint += 500
                total_reward += reward
                next_state = np.reshape(info["enemy"], [1, state_size])
                #next_state = np.append(next_state,info["time"])
                #next_state = np.reshape(next_state, [1, state_size+1])
                state = next_state
                x0 = x
            env.reset()
            done = False
            print("Total reward ", total_reward)
        env.close()

print("Total reward ", total_reward)