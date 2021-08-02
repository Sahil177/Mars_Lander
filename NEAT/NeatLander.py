##########################################################################################################
#This code will run the NEAT(NeuroEvolution of Augmenting Topologies) algorithm in order to evolve a neural network to maximimise the reward
#earnt by the neural network during a straight decent from either 10km or 200km in the mars lander environment.
#If you would like to run the algorithm for yourself you may need to install some python libraries most likely the neat library which is available via a 'pip install neat-python'
#You have control over three variables namely the NEAT Variables found on line 32
#At the end you will be presented with some summary statstics of the best nerual network and 2 graphs of the velocity profiles of the lander from a 10km and 200km decent
#Due to the randomness of the process it may not converge to a good lander on the first attempt, try again or run for more generations.
############################################################################################################
# Credit to the NEAT documentation which I used to set this up https://neat-python.readthedocs.io/en/latest/neat_overview.html
import numpy as np
import matplotlib.pyplot as plt
import random 
import neat
import os
import time


# lander variables
UNLOADED_MASS = 100.0 #KG
FUEL_MASS = 100.0 #KG
FUEL_RATE = 0.5 # KGS-1
GRAVITY = 6.673e-11 
MARS_MASS = 6.42e23 #kg
DRAG_COEF_LANDER = 1.0
LANDER_SIZE = 1.0
DRAG_COEF_CHUTE = 2.0
MARS_RADIUS = 3386000.0
MAX_THRUST = 1.5*(UNLOADED_MASS+FUEL_MASS)*GRAVITY*MARS_MASS/(MARS_RADIUS**2)
EXOSPHERE = 200_000.0
delta_t =0.1

#NEAT Variables
GENERATIONS = 50 # Number of generations you would like to run the algorithm for
STARTING_ALTITUDE = 10e3 #the performance of the nerual networks are based on a decent from this starting altitude during the 'training' process. 
REWARD_POLICY = 0 # |0 : time optmising reward policy | 1 : fuel opimising reward policy | 2 : peak acceleration optimising reward policy these can be found on line 106


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

    def action(self, throttle, stepi):
        self.move(delta_t, stepi, throttle )

class landerEnv:

    def reset(self, alt):
        self.a_max = 0
        self.landerX = lander(alt)
        obs = [self.landerX.x - MARS_RADIUS, self.landerX.v]
        return obs
    
    def step(self, action, step):
        self.landerX.action(action, step)
        self.a_max = max(self.a_max, abs(self.landerX.a))

        new_obeservation = [self.landerX.x - MARS_RADIUS, self.landerX.v]

        if REWARD_POLICY == 0:
            #time optimising reward policy
            if self.landerX.x < MARS_RADIUS:
                if self.landerX.v < -1:
                    reward = -abs(self.landerX.v)
                else:
                    reward = 200
            else:
                reward = -delta_t
        elif REWARD_POLICY == 1:
            # fuel optimising reward policy
            if self.landerX.x < MARS_RADIUS:
                if self.landerX.v < -1:
                    reward = -abs(self.landerX.v)
                else:
                    reward = self.landerX.mass - UNLOADED_MASS
            else:
                reward = -delta_t
        elif REWARD_POLICY ==2 :
            #peak acceleration optimising reward policy
            if self.landerX.x < MARS_RADIUS:
                if self.landerX.v < -1:
                    reward = -abs(self.landerX.v)
                else:
                    reward = (10 - self.a_max)
            else:
                reward = 0

        done = False

        if self.landerX.x < MARS_RADIUS:
            done = True

        return new_obeservation, reward, done


env = landerEnv()
lander10 = lander(10e3)
lander200 = lander(200e3)

def eval_genomes(genomes, config):
    for _, genome in genomes:
        i = 0
        done = False
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        ep_reward = 0
        obs = env.reset(STARTING_ALTITUDE)
        while not done:
            throttle = ((net.activate(obs))[0]+1)/2
            obs , reward, done = env.step(throttle, i)
            ep_reward += reward
        genome.fitness = ep_reward

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to x generations.
    winner = p.run(eval_genomes, GENERATIONS)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    #test winner
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #Tests model 100 times and prints result

    
    i= 0
    X = []
    V = []
    a = []
    fuel = []
    t = []
    while lander10.x > MARS_RADIUS:
        t.append(delta_t*i)
        X.append(lander10.x - MARS_RADIUS)
        V.append(abs(lander10.v))
        a.append(lander10.a)
        fuel.append(lander10.mass - UNLOADED_MASS)
        throttle = (((winner_net.activate([lander10.x - MARS_RADIUS, lander10.v]))[0])+1)/2
        lander10.action(throttle, i)
        i+=1
    
    a_abs = [abs(k) for k in a[1:]]
    pa = max(a_abs)
    
    print(f"t: {t[-1]}, x:{X[-1]}, v: {-V[-1]} fuel left: {fuel[-1]} peak acceleration: {pa}")
    plt.figure(1)
    plt.clf()
    plt.xlabel('time (s)')
    plt.grid()
    plt.plot(t, V, label='v (m)')
    plt.legend()
    plt.show()

    i= 0
    X = []
    V = []
    a = []
    fuel = []
    t = []
    while lander200.x > MARS_RADIUS:
        t.append(delta_t*i)
        X.append(lander200.x - MARS_RADIUS)
        V.append(abs(lander200.v))
        a.append(lander200.a)
        fuel.append(lander200.mass - UNLOADED_MASS)
        throttle = (((winner_net.activate([lander200.x - MARS_RADIUS, lander200.v]))[0])+1)/2
        lander200.action(throttle, i)
        i+=1
    
    a_abs = [abs(k) for k in a[1:]]
    pa = max(a_abs)

    print(f"t: {t[-1]}, x:{X[-1]}, v: {-V[-1]} fuel left: {fuel[-1]} peak acceleration: {pa}")
    plt.figure(2)
    plt.clf()
    plt.xlabel('time (s)')
    plt.grid()
    plt.plot(t, V, label='v (m)')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)