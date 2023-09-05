# Autonomous Project

## Usage

### Train DDQN Agent
```
python train_snake_DQN.py [-a DQN] [-m Models/DQN/snake_xxxx.h5] [-e 4000] [-p 1.0]
```

### Test DDQN Agent
```
python test_snake_DQN.py [-a DQN] [-m Models/DQN/snake_xxxx.h5] [-e 100] [-p 0.0]
```

### Train A2C Agent
```
python train_snake_A2C.py [-a A2C] [-ma Models/A2C/snake_xxxx_actor.h5] [-mc Models/A2C/snake_xxxx_critic.h5] [-e 100000]
```

### Test A2C Agent
```
python test_snake_A2C.py [-a A2C] [-m Models/A2C/snake_xxxx_actor.h5] [-e 100]
```

## Documentation

The documentation of the project is available in PDF and LaTeX in folder **./Report**