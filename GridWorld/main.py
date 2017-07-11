import gym
import tensorflow as tf
import os
import numpy as np
from numpy import random
from gridworld import gameEnv

from Net2 import *
#from skimage.color import rgb2gray
import cv2
from collections import deque
import random
from PIL import Image
import time
import psutil


env = gameEnv(partial=False,size=5)


batch_size = 32 #How many experiences to use for each training step.
update_freq = 1 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 0.99 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 7000 #How many episodes of game environment to train network with.
pre_train_steps = 15000 #How many steps of random actions before training begins.
max_epLength = 60 #The max allowed length of our episode.

target_freq = 1000 #How many step to upgrade the QTarget
REPLAY_MEMORY_SIZE = 20000 #Dimension of the replay Buffer




epsilon = startE
stepDrop = (startE - endE)/anneling_steps #Reduction rate per the e-greedy implementation


QTarget = Q_Net()
QMain = Q_Net()

session = tf.Session()

saver = tf.train.Saver()
session.run(tf.initialize_all_variables())

checkpoint = tf.train.get_checkpoint_state("saved_networks")
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(session, checkpoint.model_checkpoint_path)
  print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
  print("Could not find old network weights")


total_steps = 0
replay_memory = []
step_counts = []

#the two net have the same value at the beginning
session.run(QTarget.W_conv1.assign(QMain.W_conv1))
session.run(QTarget.b_conv1.assign(QMain.b_conv1))
session.run(QTarget.W_conv2.assign(QMain.W_conv2))
session.run(QTarget.b_conv2.assign(QMain.b_conv2))
session.run(QTarget.W_conv3.assign(QMain.W_conv3))
session.run(QTarget.b_conv3.assign(QMain.b_conv3))
session.run(QTarget.W_fc1.assign(QMain.W_fc1))
session.run(QTarget.b_fc1.assign(QMain.b_fc1))
session.run(QTarget.W_fc2.assign(QMain.W_fc2))
session.run(QTarget.b_fc2.assign(QMain.b_fc2))



option = input("tape 0 for train, 1 for test\n")
if(option == 0):
  # Train
  for episode in range(num_episodes):
    X_old = env.reset()

    steps = 0
    totalrew = 0
    for step in range(max_epLength):
      # Pick the next action and execute it
      action = None
      if random.random() < epsilon:
        action = np.random.randint(0,4)
      else:
        q_values = session.run(QMain.out_fc3, feed_dict={QMain.input: [X_old]})
        action = q_values.argmax()
    
      X_new, reward, done = env.step(action)
      
      # Insert in the replay_buffer
      replay_memory.append((X_old, action, reward, X_new, done))
      if len(replay_memory) > REPLAY_MEMORY_SIZE:
        replay_memory.pop(0)

      X_old = X_new
      
      # Control needed to permorf backprop after populating the replay_buffer
      if (len(replay_memory) >= pre_train_steps):
        if epsilon > endE:
          epsilon -= stepDrop
        minibatch = random.sample(replay_memory, batch_size)
        next_states = [m[3] for m in minibatch]
   
        q_values = session.run(QTarget.out_fc3, feed_dict={QTarget.input: next_states})
        max_q_values = q_values.max(axis=1)

        # Compute target Q values
        target_q = np.zeros(batch_size)
        target_action_mask = np.zeros((batch_size, 4), dtype=int)
        # This formula, is described in the alghorithm in the relation
        for i in range(batch_size):
          _, action, old_reward, _, done_ = minibatch[i]
          target_q[i] = old_reward
          if not done_:
            target_q[i] += y * max_q_values[i]
          target_action_mask[i][action] = 1

        states = [m[0] for m in minibatch]
        # Compute loss and start backProp
        _= session.run(QMain.train_op, feed_dict={
          QMain.input: states, 
          QMain.target: target_q,
          QMain.actions: target_action_mask,
        })


      total_steps += 1
      totalrew += reward
      steps += 1
      #update the weights of the target net
      if total_steps % target_freq == 0:
          session.run(QTarget.W_conv1.assign(QMain.W_conv1))
          session.run(QTarget.b_conv1.assign(QMain.b_conv1))
          session.run(QTarget.W_conv2.assign(QMain.W_conv2))
          session.run(QTarget.b_conv2.assign(QMain.b_conv2))
          session.run(QTarget.W_conv3.assign(QMain.W_conv3))
          session.run(QTarget.b_conv3.assign(QMain.b_conv3))
          session.run(QTarget.W_fc1.assign(QMain.W_fc1))
          session.run(QTarget.b_fc1.assign(QMain.b_fc1))
          session.run(QTarget.W_fc2.assign(QMain.W_fc2))
          session.run(QTarget.b_fc2.assign(QMain.b_fc2))
      if done:
        break

    step_counts.append(steps) 
    mean_steps = np.mean(step_counts[-100:])
    print("Training episode = {}, Total steps = {}, Last-100 mean steps = {}"
                                    .format(episode, total_steps, mean_steps))
    print("rew")
    if num_episodes % 1000 == 0:
      saver.save(session, 'saved_networks/' + '-dqn')
    print totalrew


else:
  # Test
  game_win = 0
  num_episodes = 200
  for i in range(num_episodes):
          X_old = env.reset()
          img = Image.fromarray(X_old,'RGB')
          img.show()
          done = False
          total_reward = 0
          j = 0
          while j < max_epLength: 
              j+=1
              action_index = session.run(QMain.out_fc3,feed_dict = {QMain.input : [X_old]})
              action_index = action_index.argmax()
              if(action_index == 0):
                print("su")
              elif (action_index == 1):
                print("giu")
              elif (action_index==2):
                print("sinistra")
              else:
                print("destra")
              
              total_steps += 1

              X_new,reward,done = env.step(action_index)

              if(reward == -1):
                print("PERSO")
                break;
              if(reward == 1):
                print("VINTO")
                game_win = game_win + 1
                break;

              X_old = X_new

              total_reward += reward
              if done == True:
                  print("PERSO")
                  break
          print total_reward
          time.sleep(5)
          for proc in psutil.process_iter():
            if proc.name()=='display':
              proc.kill()
  print("Partite vinte")
  print game_win
  print("su giocate")
  print num_episodes


