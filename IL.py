from multi_taxi import single_taxi_v0
from utils import get_taxi_location, get_passenger_locations
from collections import deque
import numpy as np

def recounstruct_path(prev, location):
    path = []
    while location:
        path.append(location)
        location = prev[location]

    path.reverse()

    return path

def bfs_pathfinding(env, start, goals):
    """
    Finds the shortest path from start to goal in the domain_map using BFS.

    Parameters:
    - env: gym.Env object.
    - start: Tuple (row, col) indicating the starting position.
    - goal: Tuple (row, col) indicating the goal position.

    Returns:
    - path: List of tuples representing the path from start to goal.
            Returns None if no path is found.
    """
    domain_map = env.unwrapped.domain_map
    rows = domain_map.map_height
    cols = domain_map.map_width

    visited = np.zeros((rows, cols), dtype=bool)
    prev = np.full((rows, cols), None, dtype=object)

    queue = deque()
    queue.append(start)
    visited[start] = True

    while queue:
        current = queue.popleft()

        if current in goals:
            return recounstruct_path(prev, current)

        row, col = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                if not visited[r, c] and not domain_map.hit_obstacle(current, (r, c)):
                    visited[r, c] = True
                    prev[r, c] = current
                    queue.append((r, c))

    return None

class BfsAgent:
    def __init__(self, env: single_taxi_v0.gym_env):
        self.env = env

        action_to_name = env.unwrapped.get_action_map()
        self.action_map = {
            (1, 0): action_to_name['south'],
            (-1, 0): action_to_name['north'],
            (0, -1): action_to_name['west'],
            (0, 1): action_to_name['east'],
            'pickup': action_to_name['pickup'],
            'refuel': action_to_name['refuel'],
        }

    def __call__(self, obs):
        taxi_loc = get_taxi_location(self.env)
        passenger_locs = get_passenger_locations(self.env)

        path = bfs_pathfinding(self.env, taxi_loc, passenger_locs)
        if len(path) == 1:
            return self.action_map['pickup']

        next_loc = path[1]
        direction = (next_loc[0] - taxi_loc[0], next_loc[1] - taxi_loc[1])
        action = self.action_map[direction]

        return action
