# DQN을 이용한 게임 에이전트 구현 - CatchGame
![TensorFlowPlayCatch](https://github.com/solaris33/CatchGame-QLearningExample-TensorFlow/blob/master/images/TensorFlowPlayCatchGame.gif)

Replay Memory를 사용한 DQN 기법을 이용하여 과일받기 게임(CatchGame)을 플레이하는 DQN 에이전트를 구현해봅시다.

## 실행법(Howo to run)

먼저 학습을 위해서 train_catch_game.py 스크립트를 실행합니다.
```
python train_catch_game.py
```

train_catch_game.py 스크립트는 파라미터를 업데이트하고 tf.train.Saver API를 이용해서 학습된 파라미터를 저장합니다.

## 학습된 DQN 에이전트로 게임 플레이하기(Play and Visualization)
터미널에서 아래 명령어로 jupyter notebook을 켜고 play_catch_game.ipynb 스크립트를 실행하면,
저장된 파리미터를 불러와서 학습된 DQN 에이전트가 게임을 플레이하는 모습을 관찰할 수 있습니다.

```
jupyter notebook
```

## References
[https://gist.github.com/EderSantana/c7222daa328f0e885093](https://gist.github.com/EderSantana/c7222daa328f0e885093)

[https://github.com/SeanNaren/TorchQLearningExample](https://github.com/SeanNaren/TorchQLearningExample)