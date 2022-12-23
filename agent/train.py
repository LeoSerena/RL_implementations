import pickle
import os

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from .agent import ConnectFourAgent
from environment.connectFourEnv import ConnectFourEnv

DEFAULT_SAVES_DIR = 'saved_models'
DEFAULT_MODEL_NAME = 'best.pt'


def train(
        num_epochs: int = 20,
        batch_size: int = 32,
        threshold: float = 0.5,
        environment: ConnectFourEnv = None,
        control_agent: ConnectFourAgent = None,
        training_agent: ConnectFourAgent = None,
        save_path: str = DEFAULT_SAVES_DIR,
        model_name: str = DEFAULT_MODEL_NAME
):
    if environment is None:
        raise ValueError('environment is null')
    if control_agent is None:
        raise ValueError('control agent is null')
    if training_agent is None:
        raise ValueError('training agent is null')

    save_model = False
    history = {}
    for e in tqdm(range(num_epochs)):
        history[e] = {
            'losses': [],
            'scores': [],
            'rewards': []
        }
        for b in range(batch_size):
            training_agent.no_grad()
            is_over = False
            batch_loss = None
            training_agent.reset_reward()
            environment.reset_state()
            while not is_over:
                sampling_dist = control_agent.actions_distribution(environment.state)
                action = sampling_dist.sample()
                is_over, reward, winner = control_agent.perform_action(action, environment)
                if is_over:
                    if winner:
                        training_agent.reward -= reward
                        history[e]['scores'].append(-1)
                    else:
                        history[e]['scores'].append(0)
                else:
                    # we generate the logits from the state
                    sampling_dist = training_agent.actions_distribution(environment.state)
                    action = sampling_dist.sample()
                    is_over, reward, winner = training_agent.perform_action(action, environment)
                    training_agent.reward += reward
                    if is_over:
                        history[e]['scores'].append(1 if winner else 0)

                    weight = training_agent.reward / batch_size
                    run_loss = - sampling_dist.log_prob(action) * weight
                    if batch_loss is None:
                        batch_loss = run_loss
                    else:
                        batch_loss += run_loss

            history[e]['losses'].append(batch_loss.item())
            history[e]['rewards'].append(training_agent.reward)

            batch_loss.backward()
            training_agent.optimizer.step()

        if is_better(history, threshold=threshold):
            save_model = True
            break

    if save_model:
        try:
            os.mkdir(save_path)
        except FileExistsError:
            pass
        training_agent.save_model(os.path.join(
            save_path,
            model_name
        ))
        X = len(os.listdir(save_path))
        with open(os.path.join(save_path, f'history_{X}.pkl'), 'wb') as f:
            pickle.dump(history, f)

    return history, save_model


def is_better(
        history,
        w: int = 10,
        threshold: float = 0.5
):
    """
    returns true if the average of the score of the w last epochs were over the threshold
    """
    M = len(history)
    m = max(0, M - w)
    if m == 0:
        return False
    scores = []
    for e in range(m, M):
        scores += history[e]['scores']
    return np.mean(scores) > threshold


def plot_history(history):
    l, r, s = [], [], []
    for i in range(len(history)):
        l.append(np.mean(history[i]['losses']))
        r.append(np.mean(history[i]['rewards']))
        s.append(np.mean(history[i]['scores']))

    fig, axs = plt.subplots(3, figsize=(16, 8))
    axs[0].set_title('Average Loss')
    axs[0].plot(l)

    axs[1].set_title('Average Reward')
    axs[1].plot(r)

    axs[2].set_title('Average Score')
    axs[2].plot(s)

    plt.show()