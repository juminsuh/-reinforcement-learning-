import numpy as np
from typing import Tuple
import pandas as pd

class GridWorldEnvironment:
    
    def __init__(self, start_point:Tuple, end_point:Tuple, gridworld_size:Tuple):
        # 시작점과 끝점을 받는다.
        self.start_point = start_point
        self.end_point = end_point if end_point != (-1,-1) else (gridworld_size[0] + end_point[0],
                                                                 gridworld_size[1] + end_point[1])

        # 그리드 월드의 규격을 받는다.
        self.width, self.height = gridworld_size

        # action dictionary
        self.action_space = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.action_space)
        self.actions = {'up':(-1,0),
                        'down':(1,0),
                        'left':(0,-1),
                        'right':(0,1) }

        # 상태 : 좌표로 나타남
        self.traces = []

        # total states
        self.total_states = []
        for x in range(self.width):
            for y in range(self.height):
                self.total_states.append((x,y))

        # reward
        self.reward = np.zeros(shape=(self.height, self.width)).tolist()
        self.reward[end_point[0]][end_point[1]] = 1

    def render(self):
        last_point=self.traces[-1]
        traces=list(set(self.traces))

        lists= [[0,0,0,0,0] for i in range(self.width)]#5개의 빈 리스트 생성

        render_df=pd.DataFrame({i:lists[i] for i in range(self.width)})

        for i in render_df.index:
            for j in render_df.columns:
                render_df.loc[i,j]='*'
        
        for trace in traces: # 어차피 index와 columns가 모두 0,1,2,3,4이기 때문에 iloc[int, int]말고 loc[index, columns]를 써도 상관 없음
            render_df.loc[trace[0], trace[1]]="X"

        render_df.loc[self.start_point[0], self.start_point[1]]="S"
        render_df.loc[self.end_point[0], self.end_point[1]]="G"
        render_df.loc[last_point[0], last_point[1]]="A"

        print(render_df)

    def get_reward(self, state, action_idx):
        next_state = self.state_after_action(state, action_idx)
        return self.reward[next_state[0]][next_state[1]]

    def state_after_action(self, state, action_idx:int):
        action = self.action_space[action_idx]
        row_movement, col_movement = self.actions[action]

        # action에 따라 에이전트 이동
        next_state = (state[0]+row_movement, state[1]+col_movement)
        next_state = self.check_boundary(next_state)

        return next_state

    def check_boundary(self, state):#그리드월드의 경계 내에서만 state가 발생하도록 유지시키는 역할
        state = list(state)
        state[0] = (0 if state[0] < 0 else self.height - 1 if state[0] > self.height - 1 else state[0])
        state[1] = (0 if state[1] < 0 else self.width - 1 if state[1] > self.width - 1 else state[1])
        return tuple(state)
