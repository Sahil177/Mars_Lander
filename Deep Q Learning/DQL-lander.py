#This code attempts to teach the mars lander to land using Deep Q learning
# Credit to https://pythonprogramming.net/ for teaching me how to go about doing this.


import numpy as np
import matplotlib.pyplot as plt
import random 
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


# lander variables
UNLOADED_MASS = 100.0 #KG
FUEL_MASS = 100.0 #KG
FUEL_RATE = 0.0 # KGS-1
GRAVITY = 6.673e-11 
MARS_MASS = 6.42e23 #kg
DRAG_COEF_LANDER = 1.0
LANDER_SIZE = 1.0
DRAG_COEF_CHUTE = 2.0
MARS_RADIUS = 3386000.0
MAX_THRUST = 1.5*(UNLOADED_MASS+FUEL_MASS)*GRAVITY*MARS_MASS/(MARS_RADIUS**2)
EXOSPHERE = 200_000.0
delta_t =5

#DQN variables
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
INPUT_SIZE =2

#environment
EPISODES = 150

# Exploration Settings
epsilon = 1
EPSILON_DECAY = 0.9954
MIN_EPSILON = 0.001

STATS_EVERY = 50

def atmospheric_density(x):
    alti = x - MARS_RADIUS
    if alti > EXOSPHERE or alti < 0.0:
        return 0.0
    else:
        return 0.017*np.exp(-alti/11000.0)

class lander:
    def __init__(self, init_altitude, velocity =0 ):
        self.xprev = None
        self.xnew = None
        self.x = init_altitude + MARS_RADIUS
        self.v = velocity
        self.mass = UNLOADED_MASS + FUEL_MASS
        self.a = None
        self.parachute_deployed = False
       
    def move(self, delta_t, step, throttle=0, para_deploy =False):
        atmos_den = atmospheric_density(self.x)
        gravity = -(GRAVITY*MARS_MASS*(self.mass))/((self.x)**2)
        dragv = -0.5*(atmos_den)*DRAG_COEF_LANDER*np.pi*(LANDER_SIZE**2)*(self.v)**2*np.sign(self.v)
        dragp = -0.5*(atmos_den)*DRAG_COEF_CHUTE*5*((2*LANDER_SIZE)**2)*(self.v)**2*np.sign(self.v)

        if self.mass > UNLOADED_MASS:
            thrust = throttle*MAX_THRUST
        else:
            thrust = 0

        if para_deploy:
            self.a  = (gravity+dragp+dragv+thrust)/ self.mass
        else:
            self.a = (gravity+dragv+thrust)/self.mass


        if step == 0:
            self.xnew = self.x + self.v*delta_t
            self.v += self.a*delta_t
        else:
            self.xnew = 2*self.x - self.xprev + self.a*delta_t**2
            self.v = (1/delta_t)*(self.xnew-self.x)


        self.xprev = self.x
        self.x = self.xnew

        if self.mass > UNLOADED_MASS:
            self.mass -= throttle*FUEL_RATE*delta_t
        else:
            self.mass = UNLOADED_MASS

    def action(self, choice, stepi):
        self.move(delta_t, stepi, throttle = choice/10.0 )
    
class landerEnv:

    def reset(self):
        self.landerX = lander(10e3)
        obs = [self.landerX.x - MARS_RADIUS, self.landerX.v]
        obs = np.asarray(obs).astype(np.float32)

        return obs
    
    def step(self, action, step):
        self.landerX.action(action, step)

        new_obeservation = [self.landerX.x - MARS_RADIUS, self.landerX.v]
        new_obeservation = np.asarray(new_obeservation).astype(np.float32)

        if self.landerX.x < MARS_RADIUS:
            reward = -(abs(self.landerX.v))
        else:
            reward = 0

        done = False

        if self.landerX.x < MARS_RADIUS:
            done = True

        return new_obeservation, reward, done


class DQNAgent:
    def __init__(self):
        
        #Main model
        self.model = self.create_model()

        #Target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen= REPLAY_MEMORY_SIZE)
        
        self.target_update_counter = 0
    
    def create_model(self):
        model = Sequential()

        model.add(Dense(16, activation='relu', input_shape=(INPUT_SIZE,) ))
        model.add(Dropout(0.2))

        model.add(Dense(11, activation='linear'))
        model.compile(loss = 'mse', optimizer= Adam(lr=0.0001), metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    def train(self, terminal_state, step):

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT*max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose= 0, shuffle = False )

        if terminal_state:
            self.target_update_counter +=1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter =0

    def get_qs(self, state):
        return self.model.predict(state.reshape(-1,INPUT_SIZE))


env = landerEnv()
ep_rewards = []
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
agent = DQNAgent()

average_rewards = []
episodes = []

for episode in range(1, EPISODES+1):
    episode_reward = 0
    step = 0
    current_state = env.reset()
    x_init = current_state[0]
    t = []
    x= []
    v = []

    
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0,11)

        new_state, reward, done = env.step(action, step)
    
        episode_reward += reward

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        t.append(step*delta_t)
        x.append(current_state[0])
        v.append(current_state[1])

        current_state = new_state
        step +=1
    
    print(f" {episode}: Release height: {x_init} t: {t[-1]}, x:{x[-1]}, v: {v[-1]}, e: {epsilon}")

    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/len(ep_rewards[-STATS_EVERY:])
        min_reward = min(ep_rewards[-STATS_EVERY:])
        max_reward = max(ep_rewards[-STATS_EVERY:])
        episodes.append(episode)
        average_rewards.append(average_reward)

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)



plt.figure(1)
plt.clf()
plt.xlabel('episode')
plt.grid()
plt.plot(episodes, average_rewards, label='reward (m)')
plt.legend()
plt.show()

'''
lander1 = lander(10e3)

X = []
V = []
a = []
t = []
i=0
while lander1.x >= MARS_RADIUS:
    t.append(delta_t*i)
    X.append(lander1.x - MARS_RADIUS)
    V.append(lander1.v)
    a.append(lander1.a)
    lander1.move(delta_t, i)
    i +=1

print(f"t: {t[-1]}, x:{X[-1]}, v: {-V[-1]}")


plt.figure(1)
plt.clf()
plt.xlabel('time (s)')
plt.grid()
plt.plot(t, a, label='a (m)')
plt.legend()
plt.show()

plt.figure(2)
plt.clf()
plt.xlabel('time (s)')
plt.grid()
plt.plot(t, V, label='v (m)')
plt.legend()
plt.show()

plt.figure(2)
plt.clf()
plt.xlabel('time (s)')
plt.grid()
plt.plot(t, X, label='x (m)')
plt.legend()
plt.show()'''
