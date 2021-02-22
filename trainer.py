from agent import *
import pandas as pd
import os
import shutil

# config
symbol, episodes = 'GOOG', 100
state_size, action_size = 30, 3
min_change_perc = 0.0044
interval = 2
max_hold = 20

# prepare models folder
if os.path.exists('models'):
    shutil.rmtree('models')
os.mkdir('models')

# log
log_file = open('models/log.txt', 'a')
def print_log(msg):
    log_file.write(msg + '\n')
    log_file.flush()
    print(msg)

# init data
data = pd.read_csv(f'data/{symbol}-Train.csv.txt', sep=' ')
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S')  # format date to datetime
data_train = data['Close']
# max_data_index = len(data.index)  # train with all data
max_data_index = 121

def get_perc_change_list(lst: list):
    return [100 * (b - a) / a for a, b in zip(lst[::1], lst[1::1])]

def get_state(index):
    return Tensor(get_perc_change_list(data_train[index-state_size-1:index]))

agent = Agent(state_size=state_size, action_size=action_size, training_mode=True, model=None)

for episode in range(episodes):
    index = state_size+1
    state = get_state(index)
    has_stock = False

    action_counter, success_counter, change_perc_sum = 0, 0, 0

    while index <= max_data_index:
        action = agent.predict(state)

        reward = 0
        if action == 0:
            if not has_stock:  # buy stock
                has_stock = True
                index_buy = index
                action_counter += 1
        elif action == 1 and has_stock:  # sell stock
            has_stock = False
            open_buy = data_train[index_buy]
            change_perc = (data_train[index] - open_buy) / open_buy
            change_perc_sum += change_perc
            if change_perc >= min_change_perc and index - index_buy <= max_hold:
                reward = 1
                success_counter += 1
            else:
                reward = -1
            # print(f'{round(change_perc, 4)}')

        reward = Tensor([reward])
        last_state = state
        index += interval
        state = get_state(index)
        agent.memory.push(last_state, action, state, reward)
        agent.ext_replay()

    # save model
    torch.save(agent.model, f'models/model_ep{episode}')

    # statistic
    if success_counter == 0 or action_counter == 0:
        success = 0
    else:
        success = round(success_counter / action_counter * 100, 2)

    if change_perc_sum == 0 or action_counter == 0:
        av_change_perc = 0
    else:
        av_change_perc = round(change_perc_sum / action_counter * 100, 4)

    msg = f'[Epoch {episode}] Actions {action_counter}, Ø-Change-Action {av_change_perc}%, Success {success}%'
    print_log(msg)
