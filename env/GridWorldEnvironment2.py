import numpy as np
from typing import Tuple

class GridWorldEnvironment:
    def __init__(self, start_point:Tuple, end_point:Tuple, gridworld_size:Tuple):
        # 시작점과 끝점을 받는다.
        self.start_point = start_point
        self.end_point = end_point

        # 그리드 월드의 규격을 받는다.
        self.width, self.height = gridworld_size

        # action dictionary
        self.action_space = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.action_space)
        self.actions = {'up':(-1,0),
                        'down':(1,0),
                        'left':(0,-1),
                        'right':(0,1) }

        # 위치 : 좌표로 나타남
        self.present_state = start_point
        self.traces = []

        # rendering
        self.grid_world = np.full(shape=(self.height, self.width), fill_value=".").tolist()

    def render(self):
        # 그리드 월드의 상태를 출력한다.

        # 지나간 흔적
        traces = list(set(self.traces)) # 중복행동을 피하기 위해서
        for trace in traces:
            self.grid_world[trace[0]][trace[1]] = "X"

        self.grid_world[self.start_point[0]][self.start_point[1]] = "S" # start point
        self.grid_world[self.end_point[0]][self.end_point[1]] = "G" # end point
        self.grid_world[self.present_state[0]][self.present_state[1]] = "A" # 현재 에이전트의 위치

        # string으로 출력한다.
        grid = ""

        for i in range(self.height):
            for j in range(self.width):
                grid += self.grid_world[i][j]+" "
            grid += "\n"

        print(grid)

    def reset(self):
        self.present_state = self.start_point
        self.grid_world = np.full(shape=(self.height, self.width), fill_value=".").tolist()
        self.traces = []
        return self.start_point

    def step(self, action_idx:int):
        '''
        에이전트의 행동에 따라 주어지는 next_state, reward, done
        '''

        # action and movement per action
        action = self.action_space[action_idx]
        row_movement, col_movement = self.actions[action]

        # action에 따라 에이전트 이동
        next_state = (self.present_state[0]+row_movement, self.present_state[1]+col_movement)
        next_state = self.check_boundary(next_state)

        #  보상 함수
        if next_state == self.end_point:
            reward = 100
            done = True
        else:
            reward = 0
            done = False

        # 현재 위치 업데이트
        self.present_state = next_state
        self.traces.append(self.present_state)

        return next_state, reward, done

    def check_boundary(self, state):
        state = list(state)
        state[0] = (0 if state[0] < 0 else self.height - 1 if state[0] > self.height - 1 else state[0])
        state[1] = (0 if state[1] < 0 else self.width - 1 if state[1] > self.width - 1 else state[1])
        return tuple(state)
