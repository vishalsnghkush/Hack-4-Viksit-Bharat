"""
Navigation Module
Implements goal definition, path planning, route memory, replanning, and global navigation logic.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from collections import deque
import math

from mapping import EnvironmentMap


class NavigationState(Enum):
    """Navigation state machine states."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    FOLLOWING_PATH = "FOLLOWING_PATH"
    BLOCKED = "BLOCKED"
    REPLANNING = "REPLANNING"
    ARRIVED = "ARRIVED"


@dataclass
class Goal:
    """Navigation goal."""
    position: Tuple[float, float]  # (x, y) in meters
    goal_type: str  # 'position', 'waypoint', 'destination'
    tolerance: float = 2.0  # meters
    priority: int = 1  # Higher = more important


@dataclass
class Waypoint:
    """A waypoint in the path."""
    position: Tuple[float, float]  # (x, y)
    speed_limit: Optional[float] = None  # m/s
    action: Optional[str] = None  # 'turn_left', 'turn_right', 'straight', etc.


@dataclass
class Path:
    """A planned path."""
    waypoints: List[Waypoint]
    total_distance: float
    estimated_time: float
    path_id: int


class PathPlanner:
    """
    Path planning using A* or similar algorithms.
    Plans paths avoiding obstacles in the map.
    """
    
    def __init__(self, map: EnvironmentMap):
        """
        Initialize path planner.
        
        Args:
            map: Environment map
        """
        self.map = map
    
    def plan_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        max_search_radius: float = 50.0
    ) -> Optional[Path]:
        """
        Plan a path from start to goal.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            max_search_radius: Maximum search radius in meters
        
        Returns:
            Planned path or None if no path found
        """
        # Check if goal is reachable
        if self.map.is_occupied(goal, radius=2.0):
            return None
        
        # Use A* algorithm
        path = self._astar(start, goal, max_search_radius)
        
        if path is None:
            return None
        
        # Convert to waypoints
        waypoints = [Waypoint(position=pos) for pos in path]
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(len(path) - 1):
            dist = math.sqrt(
                (path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2
            )
            total_distance += dist
        
        # Estimate time (assuming average speed of 10 m/s)
        estimated_time = total_distance / 10.0
        
        return Path(
            waypoints=waypoints,
            total_distance=total_distance,
            estimated_time=estimated_time,
            path_id=0
        )
    
    def _astar(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        max_radius: float
    ) -> Optional[List[Tuple[float, float]]]:
        """
        A* pathfinding algorithm.
        
        Args:
            start: Start position
            goal: Goal position
            max_radius: Maximum search radius
        
        Returns:
            List of positions forming the path
        """
        # Simplified A* implementation
        # In practice, would use a proper grid-based or sampling-based planner
        
        # Check direct path first
        if self._is_path_clear(start, goal):
            return [start, goal]
        
        # Use simplified waypoint-based planning
        # Sample intermediate waypoints
        num_waypoints = 5
        waypoints = [start]
        
        for i in range(1, num_waypoints):
            t = i / num_waypoints
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            
            # Check if waypoint is clear
            if not self.map.is_occupied((x, y), radius=1.5):
                waypoints.append((x, y))
        
        waypoints.append(goal)
        
        # Verify path is clear
        for i in range(len(waypoints) - 1):
            if not self._is_path_clear(waypoints[i], waypoints[i+1]):
                # Try to find alternative
                alt_waypoint = self._find_alternative_waypoint(
                    waypoints[i], waypoints[i+1]
                )
                if alt_waypoint:
                    waypoints.insert(i+1, alt_waypoint)
                else:
                    return None  # Path blocked
        
        return waypoints
    
    def _is_path_clear(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        step_size: float = 0.5
    ) -> bool:
        """
        Check if a straight path is clear of obstacles.
        
        Args:
            start: Start position
            end: End position
            step_size: Step size for checking
        
        Returns:
            True if path is clear
        """
        dist = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        num_steps = int(dist / step_size) + 1
        
        for i in range(num_steps + 1):
            t = i / num_steps if num_steps > 0 else 0
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            
            if self.map.is_occupied((x, y), radius=1.0):
                return False
        
        return True
    
    def _find_alternative_waypoint(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Find an alternative waypoint to avoid obstacle.
        
        Args:
            start: Start position
            end: End position
        
        Returns:
            Alternative waypoint or None
        """
        # Try perpendicular offsets
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length < 1e-6:
            return None
        
        # Perpendicular vector
        perp_x = -dy / length
        perp_y = dx / length
        
        # Try offsets
        for offset in [2.0, 3.0, 4.0, -2.0, -3.0, -4.0]:
            alt_x = (start[0] + end[0]) / 2 + offset * perp_x
            alt_y = (start[1] + end[1]) / 2 + offset * perp_y
            
            if not self.map.is_occupied((alt_x, alt_y), radius=1.5):
                return (alt_x, alt_y)
        
        return None


class RouteMemory:
    """
    Remembers past paths and routes for reuse.
    """
    
    def __init__(self, max_routes: int = 10):
        """
        Initialize route memory.
        
        Args:
            max_routes: Maximum number of routes to remember
        """
        self.routes: deque = deque(maxlen=max_routes)
        self.route_history: List[Dict] = []
    
    def save_route(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        path: Path,
        success: bool = True
    ):
        """
        Save a route to memory.
        
        Args:
            start: Start position
            goal: Goal position
            path: Planned path
            success: Whether the route was successfully followed
        """
        route_entry = {
            'start': start,
            'goal': goal,
            'path': path,
            'success': success,
            'timestamp': 0.0  # Would use actual timestamp
        }
        self.routes.append(route_entry)
        self.route_history.append(route_entry)
    
    def find_similar_route(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        threshold: float = 5.0
    ) -> Optional[Path]:
        """
        Find a similar route from memory.
        
        Args:
            start: Start position
            goal: Goal position
            threshold: Distance threshold for similarity
        
        Returns:
            Similar path or None
        """
        for route in self.routes:
            start_dist = math.sqrt(
                (route['start'][0] - start[0])**2 + (route['start'][1] - start[1])**2
            )
            goal_dist = math.sqrt(
                (route['goal'][0] - goal[0])**2 + (route['goal'][1] - goal[1])**2
            )
            
            if start_dist < threshold and goal_dist < threshold and route['success']:
                return route['path']
        
        return None


class GlobalNavigator:
    """
    Global navigation logic that coordinates path planning, route memory, and replanning.
    """
    
    def __init__(self, map: EnvironmentMap):
        """
        Initialize global navigator.
        
        Args:
            map: Environment map
        """
        self.map = map
        self.planner = PathPlanner(map)
        self.route_memory = RouteMemory()
        
        self.current_goal: Optional[Goal] = None
        self.current_path: Optional[Path] = None
        self.current_waypoint_index: int = 0
        
        self.navigation_state = NavigationState.IDLE
        self.current_position: Tuple[float, float] = (0.0, 0.0)
        
        self.replan_count = 0
        self.max_replans = 5
    
    def set_goal(self, goal: Goal):
        """
        Set navigation goal.
        
        Args:
            goal: Navigation goal
        """
        self.current_goal = goal
        self.navigation_state = NavigationState.PLANNING
        self.replan_count = 0
    
    def update_position(self, position: Tuple[float, float]):
        """
        Update current position.
        
        Args:
            position: Current position (x, y)
        """
        self.current_position = position
    
    def plan(self) -> bool:
        """
        Plan path to current goal.
        
        Returns:
            True if path found
        """
        if self.current_goal is None:
            return False
        
        # Check route memory first
        cached_path = self.route_memory.find_similar_route(
            self.current_position,
            self.current_goal.position
        )
        
        if cached_path:
            self.current_path = cached_path
            self.current_waypoint_index = 0
            self.navigation_state = NavigationState.FOLLOWING_PATH
            return True
        
        # Plan new path
        path = self.planner.plan_path(
            self.current_position,
            self.current_goal.position
        )
        
        if path:
            self.current_path = path
            self.current_waypoint_index = 0
            self.navigation_state = NavigationState.FOLLOWING_PATH
            
            # Save to memory
            self.route_memory.save_route(
                self.current_position,
                self.current_goal.position,
                path,
                success=True
            )
            return True
        else:
            self.navigation_state = NavigationState.BLOCKED
            return False
    
    def check_path_blocked(self) -> bool:
        """
        Check if current path is blocked.
        
        Returns:
            True if path is blocked
        """
        if self.current_path is None:
            return False
        
        # Check next waypoint
        if self.current_waypoint_index < len(self.current_path.waypoints):
            next_waypoint = self.current_path.waypoints[self.current_waypoint_index]
            
            # Check if path to next waypoint is clear
            if not self.planner._is_path_clear(
                self.current_position,
                next_waypoint.position
            ):
                return True
        
        return False
    
    def replan(self) -> bool:
        """
        Replan path when blocked.
        
        Returns:
            True if replanning successful
        """
        if self.current_goal is None:
            return False
        
        if self.replan_count >= self.max_replans:
            self.navigation_state = NavigationState.BLOCKED
            return False
        
        self.navigation_state = NavigationState.REPLANNING
        self.replan_count += 1
        
        # Try to plan new path
        success = self.plan()
        
        if success:
            self.navigation_state = NavigationState.FOLLOWING_PATH
        else:
            self.navigation_state = NavigationState.BLOCKED
        
        return success
    
    def update(self) -> Dict:
        """
        Update navigation state.
        
        Returns:
            Dictionary with navigation status
        """
        # Check if goal reached
        if self.current_goal:
            dist_to_goal = math.sqrt(
                (self.current_position[0] - self.current_goal.position[0])**2 +
                (self.current_position[1] - self.current_goal.position[1])**2
            )
            
            if dist_to_goal < self.current_goal.tolerance:
                self.navigation_state = NavigationState.ARRIVED
                self.current_path = None
                self.current_goal = None
        
        # Check if path is blocked
        if self.navigation_state == NavigationState.FOLLOWING_PATH:
            if self.check_path_blocked():
                self.navigation_state = NavigationState.BLOCKED
        
        # Get next waypoint
        next_waypoint = None
        if self.current_path and self.current_waypoint_index < len(self.current_path.waypoints):
            next_waypoint = self.current_path.waypoints[self.current_waypoint_index]
            
            # Check if waypoint reached
            if next_waypoint:
                dist = math.sqrt(
                    (self.current_position[0] - next_waypoint.position[0])**2 +
                    (self.current_position[1] - next_waypoint.position[1])**2
                )
                
                if dist < 2.0:  # Waypoint tolerance
                    self.current_waypoint_index += 1
        
        return {
            'state': self.navigation_state.value,
            'has_goal': self.current_goal is not None,
            'has_path': self.current_path is not None,
            'next_waypoint': next_waypoint.position if next_waypoint else None,
            'waypoint_index': self.current_waypoint_index,
            'total_waypoints': len(self.current_path.waypoints) if self.current_path else 0,
            'replan_count': self.replan_count
        }
    
    def get_next_command(self) -> Optional[Dict]:
        """
        Get next navigation command.
        
        Returns:
            Navigation command or None
        """
        if self.current_path is None or self.current_waypoint_index >= len(self.current_path.waypoints):
            return None
        
        next_waypoint = self.current_path.waypoints[self.current_waypoint_index]
        
        # Calculate heading to waypoint
        dx = next_waypoint.position[0] - self.current_position[0]
        dy = next_waypoint.position[1] - self.current_position[1]
        heading = math.atan2(dy, dx)
        
        # Calculate distance
        distance = math.sqrt(dx*dx + dy*dy)
        
        return {
            'heading': heading,
            'distance': distance,
            'waypoint': next_waypoint.position,
            'speed_limit': next_waypoint.speed_limit
        }
