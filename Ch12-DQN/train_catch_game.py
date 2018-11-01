# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import math
import os

# 학습에 필요한 설정값들을 선언합니다.
epsilon = 1                 		# epsilon-Greedy 기법에 사용할 최초의 epsilon값
epsilonMinimumValue = 0.001 		# epsilon의 최소값 (이 값 이하로 Decay하지 않습니다)
num_actions = 3               	# 에이전트가 취할 수 있는 행동의 개수 - (좌로 움직이기, 가만히 있기, 우로 움직이기)
num_epochs = 2000 					    # 학습에 사용할 반복횟수
hidden_size = 128 					    # 히든레이어의 노드 개수
maxMemory = 500 					      # Replay Memory의 크기
batch_size = 50 					      # 학습에 사용할 배치 개수
gridSize = 10 						      # 에이전트가 플레이하는 게임 화면 크기 (10x10 grid)
state_size = gridSize * gridSize 	# 게임 환경의 현재상태 (10x10 grid)
discount = 0.9 						        # Discount Factor \gamma
learning_rate = 0.2					      # 러닝 레이트

# s와 e사이의 랜덤한 값을 리턴하는 유틸리티 함수를 정의합니다.
def randf(s, e):
  return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;

# DQN 모델을 정의합니다.
# 100(현재 상태 - 10x10 Grid) -> 128 -> 128 -> 3(예측된 각 행동의 Q값)
def build_DQN(x):
  W1 = tf.Variable(tf.truncated_normal(shape=[state_size, hidden_size], stddev=1.0 / math.sqrt(float(state_size))))
  b1 = tf.Variable(tf.truncated_normal(shape=[hidden_size], stddev=0.01))  
  H1_output = tf.nn.relu(tf.matmul(x, W1) + b1)
  W2 = tf.Variable(tf.truncated_normal(shape=[hidden_size, hidden_size],stddev=1.0 / math.sqrt(float(hidden_size))))
  b2 = tf.Variable(tf.truncated_normal(shape=[hidden_size], stddev=0.01))
  H2_output = tf.nn.relu(tf.matmul(H1_output, W2) + b2)
  W3 = tf.Variable(tf.truncated_normal(shape=[hidden_size, num_actions],stddev=1.0 / math.sqrt(float(hidden_size))))
  b3 = tf.Variable(tf.truncated_normal(shape=[num_actions], stddev=0.01))
  output_layer = tf.matmul(H2_output, W3) + b3

  return tf.squeeze(output_layer)

# 인풋 화면 이미지와 타겟 Q값을 받기 위한 플레이스 홀더를 선언합니다.
x = tf.placeholder(tf.float32, shape=[None, state_size])
y = tf.placeholder(tf.float32, shape=[None, num_actions])

# DQN 모델을 선언하고 예측결과를 리턴받습니다.
y_pred = build_DQN(x)

# MSE 손실 함수와 옵티마이저를 정의합니다.
loss = tf.reduce_sum(tf.square(y-y_pred)) / (2*batch_size)  # MSE 손실 함수
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# CatchGame을 수행하는 Environment를 구현합니다.
class CatchEnvironment():
  # 상태의 초기값을 지정합니다.
  def __init__(self, gridSize):
    self.gridSize = gridSize
    self.state_size = self.gridSize * self.gridSize
    self.state = np.empty(3, dtype = np.uint8) 

  # 관찰 결과를 리턴합니다.
  def observe(self):
    canvas = self.drawState()
    canvas = np.reshape(canvas, (-1,self.state_size))
    return canvas

  # 현재 상태(fruit, basket)를 화면에 출력합니다.
  def drawState(self):
    canvas = np.zeros((self.gridSize, self.gridSize))
    # fruit를 화면에 그립니다.
    canvas[self.state[0]-1, self.state[1]-1] = 1  
    # basket을 화면에 그립니다. 
    canvas[self.gridSize-1, self.state[2] -1 - 1] = 1
    canvas[self.gridSize-1, self.state[2] -1] = 1
    canvas[self.gridSize-1, self.state[2] -1 + 1] = 1    
    return canvas        

  # 게임을 초기 상태로 리셋합니다.
  def reset(self): 
    initialFruitColumn = random.randrange(1, self.gridSize + 1)
    initialBucketPosition = random.randrange(2, self.gridSize + 1 - 1)
    self.state = np.array([1, initialFruitColumn, initialBucketPosition]) 
    return self.getState()

  # 현재 상태를 불러옵니다.
  def getState(self):
    stateInfo = self.state
    fruit_row = stateInfo[0]
    fruit_col = stateInfo[1]
    basket = stateInfo[2]
    return fruit_row, fruit_col, basket

  # 에이전트가 취한 행동에 대한 보상을 줍니다.
  def getReward(self):
    fruitRow, fruitColumn, basket = self.getState()
    # 만약 fruit가 바닥에 닿았을 때
    if (fruitRow == self.gridSize - 1):  
      # basket이 fruit을 받아내면 1의 reward를 줍니다.
      if (abs(fruitColumn - basket) <= 1): 
        return 1
      # fruit를 받아내지 못하면 -1의 reward를 줍니다.
      else:
        return -1
    # fruit가 바닥에 닿지 않은 중립적인 상태는 0의 reward를 줍니다.
    else:
      return 0

  # 게임이 끝났는지를 체크합니다.(fruit가 바닥에 닿으면 한게임이 종료됩니다.)
  def isGameOver(self):
    if (self.state[0] == self.gridSize - 1): 
      return True 
    else: 
      return False 

  # action(좌로 한칸 이동, 제자리, 우로 한칸이동)에 따라 basket의 위치를 업데이트합니다.
  def updateState(self, action):
    move = 0
    if (action == 0):
      move = -1
    elif (action == 1):
      move = 0
    elif (action == 2):
      move = 1
    fruitRow, fruitColumn, basket = self.getState()
    newBasket = min(max(2, basket + move), self.gridSize - 1) # min/max는 basket이 grid밖으로 벗어나는것을 방지합니다.
    fruitRow = fruitRow + 1  # fruit는 매 행동을 취할때마다 1칸씩 아래로 떨어집니다. 
    self.state = np.array([fruitRow, fruitColumn, newBasket])

  # 행동을 취합니다. 0 : 왼쪽으로 이동, 1 : 가만히 있기, 2 : 오른쪽으로 이동
  def act(self, action):
    self.updateState(action)
    reward = self.getReward()
    gameOver = self.isGameOver()
    return self.observe(), reward, gameOver, self.getState()


# Replay Memory를 class로 정의합니다.
class ReplayMemory:
  def __init__(self, gridSize, maxMemory, discount):
    self.maxMemory = maxMemory
    self.gridSize = gridSize
    self.state_size = self.gridSize * self.gridSize
    self.discount = discount
    canvas = np.zeros((self.gridSize, self.gridSize))
    canvas = np.reshape(canvas, (-1,self.state_size))
    self.inputState = np.empty((self.maxMemory, 100), dtype = np.float32)
    self.actions = np.zeros(self.maxMemory, dtype = np.uint8)
    self.nextState = np.empty((self.maxMemory, 100), dtype = np.float32)
    self.gameOver = np.empty(self.maxMemory, dtype = np.bool)
    self.rewards = np.empty(self.maxMemory, dtype = np.int8) 
    self.count = 0
    self.current = 0

  # 경험을 Replay Memory에 저장합니다.
  def remember(self, currentState, action, reward, nextState, gameOver):
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.inputState[self.current, ...] = currentState
    self.nextState[self.current, ...] = nextState
    self.gameOver[self.current] = gameOver
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.maxMemory

  def getBatch(self, y_pred, batch_size, num_actions, state_size, sess, X):
    # 취할 수 있는 가장 큰 배치 사이즈를 선택합니다. (학습 초기에는 batch_size만큼의 기억이 없습니다.)
    memoryLength = self.count
    chosenBatchSize = min(batch_size, memoryLength)

    # 인풋 데이터와 타겟데이터를 선언합니다. 
    inputs = np.zeros((chosenBatchSize, state_size))
    targets = np.zeros((chosenBatchSize, num_actions))

    # 배치안의 값을 설정합니다.
    for i in range(chosenBatchSize):
      # 배치에 포함될 기억을 랜덤으로 선택합니다.
      randomIndex = random.randrange(0, memoryLength)
      # 현재 상태와 Q값을 불러옵니다.
      current_inputState = np.reshape(self.inputState[randomIndex], (1, 100))
      target = sess.run(y_pred, feed_dict={X: current_inputState})
      
      # 현재 상태 바로 다음 상태를 불러오고 다음 상태에서 취할수 있는 가장 큰 Q값을 계산합니다.
      current_nextState = np.reshape(self.nextState[randomIndex], (1, 100))
      nextStateQ = sess.run(y_pred, feed_dict={X: current_nextState})      
      nextStateMaxQ = np.amax(nextStateQ)
      # 만약 게임오버라면 reward로 Q값을 업데이트하고 
      if (self.gameOver[randomIndex] == True):
        target[self.actions[randomIndex]] = self.rewards[randomIndex]
      # 게임오버가 아니라면 타겟 Q값(최적의 Q값)을 아래 수식을 이용해서 계산합니다.
      # Q* = reward + discount(gamma) * max_a' Q(s',a')
      else:
        target[self.actions[randomIndex]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

      # 인풋과 타겟 데이터에 값을 지정합니다.
      inputs[i] = current_inputState
      targets[i] = target

    return inputs, targets

# 학습을 진행하는 main 함수를 정의합니다.
def main(_):
  print("트레이닝을 시작합니다.")

  # 게임 플레이 환경을 선언합니다.
  env = CatchEnvironment(gridSize)

  # Replay Memory를 선언합니다.
  memory = ReplayMemory(gridSize, maxMemory, discount)

  # 학습된 파라미터를 저장하기 위한 tf.train.Saver를 선언합니다.
  saver = tf.train.Saver()
  
  winCount = 0
  with tf.Session() as sess:   
    # 변수들의 초기값을 할당합니다.
    sess.run(tf.global_variables_initializer())

    for i in range(num_epochs+1):
      # 환경을 초기화합니다.
      err = 0
      env.reset()
     
      isGameOver = False

      # 최초의 상태를 불러옵니다.
      currentState = env.observe()
            
      while (isGameOver != True):
        action = -9999  # Q값을 초기화합니다.
        # epsilon-Greedy 기법에 따라 랜덤한 행동을 할지 최적의 행동을 할지를 결정합니다.
        global epsilon
        if (randf(0, 1) <= epsilon):
          # epsilon 확률만큼 랜덤한 행동을 합니다.
          action = random.randrange(0, num_actions)
        else:          
          # (1-epsilon) 확률만큼 최적의 행동을 합니다.
          # 현재 상태를 DQN의 인풋으로 넣어서 예측된 최적의 Q(s,a)값들을 리턴받습니다.
          q = sess.run(y_pred, feed_dict={x: currentState})   
          # Q(s,a)가 가장 높은 행동을 선택합니다.
          action = q.argmax()

        # epsilon값을 0.9999만큼 Decay합니다.
        if (epsilon > epsilonMinimumValue):
          epsilon = epsilon * 0.999
        
        # 에이전트가 행동을 하고 다음 보상과 다음 상태에 대한 정보를 리턴 받습니다.
        nextState, reward, gameOver, stateInfo = env.act(action)
            
        # 만약 과일을 제대로 받아냈다면 승리 횟수를 1 올립니다.
        if (reward == 1):
          winCount = winCount + 1

        # 에이전트가 수집한 정보를 Replay Memory에 저장합니다.
        memory.remember(currentState, action, reward, nextState, gameOver)
        
        # 현재 상태를 다음 상태로 업데이트하고 GameOver유무를 체크합니다.
        currentState = nextState
        isGameOver = gameOver
                
        # Replay Memory로부터 학습에 사용할 Batch 데이터를 불러옵니다.
        inputs, targets = memory.getBatch(y_pred, batch_size, num_actions, state_size, sess, x)
        
        # 최적화를 수행하고 손실함수를 리턴받습니다.
        _, loss_print = sess.run([optimizer, loss], feed_dict={x: inputs, y: targets})  
        err = err + loss_print

      print("반복(Epoch): %d, 에러(err): %.4f, 승리횟수(Win count): %d, 승리비율(Win ratio): %.4f" % (i, err, winCount, float(winCount)/float(i+1)*100))
    # 학습이 모두 끝나면 파라미터를 지정된 경로에 저장합니다.
    print("트레이닝 완료")
    save_path = saver.save(sess, os.getcwd()+"/model.ckpt")
    print("%s 경로에 파라미터가 저장되었습니다" % save_path)

if __name__ == '__main__':
  # main 함수를 호출합니다.
  tf.app.run()