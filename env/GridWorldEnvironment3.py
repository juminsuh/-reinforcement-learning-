import numpy as np
from typing import Tuple

class GridWorldEnvironment:
    def __init__(self, start_point:Tuple, end_point:Tuple, gridworld_size:Tuple):
        # 시작점과 끝점을 받는다.
        self.start_point = start_point
        self.end_point = end_point

        # 그리드 월드의 규격을 받는다.
        self.height, self.width = gridworld_size

        # action dictionary
        self.action_space = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.action_space)
        self.actions = {'up':(-1,0),
                        'down':(1,0),
                        'left':(0,-1),
                        'right':(0,1) }

        # 위치 : 좌표로 나타남
        self.present_coords = start_point
        self.traces = []

        self.state_len = 3 # 장애물이 없는 상황이라서 (거리차이 x,y, done 유무)

    def render(self):
        # 그리드 월드의 상태를 출력한다.
        self.grid_world = np.full(shape=(self.height, self.width), fill_value=".").tolist()

        # 지나간 흔적
        traces = list(set(self.traces)) # 중복행동을 피하기 위해서
        for trace in traces:
            self.grid_world[trace[0]][trace[1]] = "X"

        self.grid_world[self.start_point[0]][self.start_point[1]] = "S" # start point
        self.grid_world[self.end_point[0]][self.end_point[1]] = "G" # end point
        self.grid_world[self.present_coords[0]][self.present_coords[1]] = "A" # 현재 에이전트의 위치

        # string으로 출력한다.
        grid = ""

        for i in range(self.height):
            for j in range(self.width):
                grid += self.grid_world[i][j]+" "
            grid += "\n"

        print(grid)

    def reset(self):
        self.present_coords = self.start_point
        self.traces = []
        return self.get_state(self.present_coords)

    def step(self, action_idx:int):
        '''
        에이전트의 행동에 따라 주어지는 next_coords, reward, done
        '''

        # action and movement per action
        action = self.action_space[action_idx]
        row_movement, col_movement = self.actions[action]

        # action에 따라 에이전트 이동
        next_coords = (self.present_coords[0]+row_movement, self.present_coords[1]+col_movement)
        next_coords = self.check_boundary(next_coords)

        #  보상 함수
        if next_coords == self.end_point:
            reward = 100
            done = True
        else:
            reward = 0
            done = False

        # 현재 위치 업데이트
        self.present_coords = next_coords
        self.traces.append(self.present_coords)

        state = self.get_state(self.present_coords)

        return state, reward, done

    def get_state(self, present_coords):

        states = list()

        states.append(self.end_point[0] - present_coords[0])
        states.append(self.end_point[1] - present_coords[1])

        if present_coords == self.end_point:
            states.append(1)
        else:
            states.append(0)

        return states


    def check_boundary(self, coords):
        coords = list(coords)
        coords[0] = (0 if coords[0] < 0 else self.height - 1 if coords[0] > self.height - 1 else coords[0])
        coords[1] = (0 if coords[1] < 0 else self.width - 1 if coords[1] > self.width - 1 else coords[1])
        return tuple(coords)
