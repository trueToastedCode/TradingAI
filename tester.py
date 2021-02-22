from agent import *
import pandas as pd
import matplotlib.pyplot as plt

# config
symbol, start, stop = 'GOOG', 40, 41
state_size, action_size = 30, 3
plt_show = True
min_change_perc = 0
max_hold = 20

# init data
data = pd.read_csv(f'data/{symbol}-Train.csv.txt', sep=' ')
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S')  # format date to datetime
data_train = data['Close']
max_data_index = len(data.index)  # train with all data
# max_data_index = 121

if plt_show:
    fig, ax = plt.subplots()
    plt.plot(data_train[:max_data_index])

def get_perc_change_list(lst: list):
    return [100 * (b - a) / a for a, b in zip(lst[::1], lst[1::1])]

def get_state(index):
    return Tensor(get_perc_change_list(data_train[index-state_size-1:index]))

for episode in range(start, stop):
    agent = Agent(state_size=state_size, action_size=action_size, training_mode=False, model=torch.load(f'models/model_ep{episode}'))
    index = state_size + 1
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
                open_buy_index = index
        elif action == 1 and has_stock:  # sell stock
            has_stock = False
            open_buy = data_train[open_buy_index]
            change_perc = (data_train[index] - open_buy) / open_buy
            change_perc_sum += change_perc
            if change_perc >= min_change_perc and index - open_buy_index <= max_data_index:
                success_counter += 1
            if plt_show:
                ax.axvspan(open_buy_index, index, ymin=0, ymax=1, alpha=0.5, color='green')

        index += 1
        state = get_state(index)

    # statistic
    if success_counter == 0 or action_counter == 0:
        success = 0
    else:
        success = round(success_counter / action_counter * 100, 2)

    if change_perc_sum == 0 or action_counter == 0:
        av_change_perc = 0
    else:
        av_change_perc = round(change_perc_sum / action_counter * 100, 4)

    print(f'[Epoch {episode}] Actions {action_counter}, Ø-Change-Action {av_change_perc}%, Success {success}%')

    if plt_show:
        plt.show()
