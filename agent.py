import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

# region use always cuda if available
# if torch.cuda.is_available():
#     FloatTensor = torch.cuda.FloatTensor
#     LongTensor = torch.cuda.LongTensor
#     ByteTensor = torch.cuda.ByteTensor
#     Tensor = FloatTensor
# else:
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor
# endregion


# contains transitions [(s, a, n ,r), ...] <-> state, action, next state, reward
class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []  # all transitions
        self.pos = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # add entry if capacity left
        self.memory[self.pos] = (state, action, next_state, reward)
        self.pos = (self.pos + 1) % self.capacity  # next pos in row, restart at beginning after max pos

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # return batch_size random transitions

    def __len__(self):
        return len(self.memory)  # return current len of filled memory (not directly capacity)


class Model(nn.Module):
    def __init__(self, state_size, action_size):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(state_size, 64)
        self.lin2 = nn.Linear(32, 16)
        self.lin3 = nn.Linear(16, 8)
        self.lin4 = nn.Linear(8, action_size)

    def forward(self, x):
        x = torch.sigmoid(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))
        return self.lin4(x)


# used for prediction and learning, optimize attributes
class Agent:
    def __init__(self, state_size, action_size, training_mode, model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.training_mode = training_mode
        self.model = model or Model(state_size, action_size)
        # if model is None and torch.cuda.is_available():
        #     self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = Memory(2000)  # capacity of memory
        self.done = 0  # counter for acts
        self.eps_start = 0.999  # act random factor at exploration
        self.eps_end = 0.01  # act random factor at exploitation
        self.eps_steps = 250  # discount step factor from exploration to exploitation
        self.batch_size = 128
        self.gamma = 0.95  # how important are future rewards

    def predict(self, state):
        self.done += 1
        if not self.training_mode:  # use always model to predict if not training
            return self.model(state).type(FloatTensor).max(0)[1]
        epsilon = random.random()  # float 0-1
        # transition from exploration to exploitation
        threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.done / self.eps_steps)
        if epsilon > threshold:  # use model (at beginning <-> comes down to optimizations)
            return self.model(state).type(FloatTensor).max(0)[1]
        return LongTensor([random.randint(0, self.action_size - 1)])  # random action

    def ext_replay(self):
        if len(self.memory) < self.batch_size:
            return  # not enough data (predict more)
        mini_batch = self.memory.sample(self.batch_size)  # [(s, a, n ,r), ...] <-> state, action, next state, reward
        for state, action, next_state, reward in mini_batch:
            action_value = self.model(state)[action].view(1)
            if next_state is None:
                target_action_value = reward
            else:
                target_action_value = (self.gamma * self.model(next_state).max().item()) + reward
            loss = F.smooth_l1_loss(action_value, target_action_value)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
