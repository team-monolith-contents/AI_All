import numpy as np
from ipywidgets import IntText, IntSlider, Button, HBox, Output
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import gym

class Agent():
    def __init__(self, map):
        self.__env = gym.make('FrozenLake-v1', desc=map, render_mode='rgb_array', is_slippery=False)
        self.__env.reset()
        
        self.__action_size = self.__env.action_space.n
        self.__observation_size = self.__env.observation_space.n
        self.__Q = np.zeros([self.__observation_size, self.__action_size])
        
        self.__epsilon = 1
        self.__alpha = 0.95
        self.__gamma = 0.8
        self.__lamb = 0.9
        self.__epsilon_update = 1000
        self.__episode_rewards = []
        self.__show_interval = 1000
        self.__plot_interval = 1000
        
        self.__episode = IntSlider(
            value=0,
            min=0,
            max=100000,
            step=1,
            disabled = True
        )
        
        self.__N_episode = IntText(
            value=10000,
            description='학습횟수:',
            disabled=False
        )
        
        self.__learn_button = Button(
            description='학습하기',
            disabled=False
        )
        
        self.__test_button = Button(
            description='테스트하기',
            disabled=False
        )
            
        self.__out = Output()
        display(HBox([self.__learn_button, self.__N_episode, self.__episode]), self.__test_button, self.__out)
        self.__learn_button.on_click(self.__learn)
        self.__test_button.on_click(self.__test)

    def __epsilon_greedy(self, state):
        if np.random.rand() > self.__epsilon:
            return np.argmax(self.__Q[state, :])
        else:
            return self.__env.action_space.sample()
            
    def __learn(self, button):
        for _ in range(self.__N_episode.value):
            self.__env.reset()
            state = 0
            rewards = []
            done = False

            if self.__episode.value % self.__epsilon_update == 0:
                self.__epsilon *= self.__lamb

            if self.__episode.value % self.__plot_interval == 0:
                    self.__show()
                    with self.__out:
                        print(self.__Q)

            while not done:
                action = self.__epsilon_greedy(state)
                next_state, reward, done, _, info = self.__env.step(action)
                self.__Q[state][action] += self.__alpha * (
                            reward + self.__gamma * self.__Q[next_state].max() - self.__Q[state][action])
                state = next_state
                rewards.append(reward)

                if self.__episode.value % self.__show_interval == 0:
                    self.__show()

            try:
                self.__episode_rewards[(self.__episode.value)//self.__plot_interval] += sum(rewards)/len(rewards)/self.__plot_interval
            except:
                self.__episode_rewards.append(sum(rewards)/len(rewards)/self.__plot_interval)

            self.__episode.value += 1
                
    def __show(self):
        with self.__out:
            screen = self.__env.render()
            clear_output(wait=True)
            plt.clf()
            
            plt.subplot(1,2,1)
            plt.plot(np.arange(len(self.__episode_rewards)+1) * self.__plot_interval, [0] + (self.__episode_rewards), marker = 'o')
            plt.subplot(1,2,2)
            plt.imshow(screen) # screen 배열을 이미지로 출력합니다.
            
            plt.show()

    def __test(self, button):
        with self.__out:
            self.__env.reset()
            cur_state=0
            done = False
            self.__show()
            while not done:
                next_state, reward, done, info,_ = self.__env.step(self.__Q[cur_state].argmax())
                cur_state = next_state
                with self.__out:
                    print("테스트중입니다....")
                self.__show()