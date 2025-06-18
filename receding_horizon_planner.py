# receding_horizon_planner.py

import heapq
import numpy as np

class AStarPlanner:
    def __init__(self, x_range, y_range, resolution, obstacles=None):
        self.minx, self.maxx = x_range
        self.miny, self.maxy = y_range
        self.resolution = resolution
        self.obstacles = obstacles or []  # list of (x, y, radius)

    def plan(self, start, goals):
        """
        Plan a path from `start` through each point in `goals` (in order),
        returning a flat list of (x,y) waypoints.
        """
        path = []
        curr = start
        for goal in goals:
            segment = self._astar(curr, goal)
            if segment:
                # drop last point to avoid duplicates
                path += segment[:-1]
                curr = goal
        if goals:
            path.append(goals[-1])
        return path

    def _astar(self, start, goal):
        def to_idx(pt):
            return (
                int((pt[0] - self.minx) / self.resolution),
                int((pt[1] - self.miny) / self.resolution)
            )
        def to_pt(idx):
            return (
                idx[0] * self.resolution + self.minx,
                idx[1] * self.resolution + self.miny
            )

        sx, sy = to_idx(start)
        gx, gy = to_idx(goal)

        open_set = [(0, (sx, sy))]
        came_from = {}
        g_score = { (sx, sy): 0 }

        def h(a, b):
            return np.hypot(a[0]-b[0], a[1]-b[1])

        directions = [
            (1,0),(-1,0),(0,1),(0,-1),
            (1,1),(1,-1),(-1,1),(-1,-1)
        ]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == (gx, gy):
                # reconstruct
                path = []
                while current in came_from:
                    path.append(to_pt(current))
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in directions:
                neighbor = (current[0]+dx, current[1]+dy)
                pt = to_pt(neighbor)
                # bounds
                if not (self.minx <= pt[0] <= self.maxx and
                        self.miny <= pt[1] <= self.maxy):
                    continue
                # obstacle check
                if any(np.hypot(pt[0]-ox, pt[1]-oy) < r
                       for ox, oy, r in self.obstacles):
                    continue

                tentative_g = g_score[current] + h(current, neighbor)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + h(neighbor, (gx, gy))
                    heapq.heappush(open_set, (f, neighbor))

        return None