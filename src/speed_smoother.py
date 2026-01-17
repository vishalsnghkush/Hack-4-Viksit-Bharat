"""
Speed Smoother Module
Rule-based speed smoothing controller that minimizes acceleration spikes.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class DrivingState(Enum):
    """Driving state machine states."""
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    DANGER = "DANGER"


@dataclass
class ObstacleInfo:
    """Information about detected obstacles."""
    relative_speed: float  # Positive = approaching
    closing_trend: str  # 'approaching', 'receding', 'stable'
    distance_estimate: float  # Normalized distance (0-1, 1 = far, 0 = close)
    is_traffic_light: bool
    traffic_light_state: Optional[str]  # 'red', 'yellow', 'green', None


@dataclass
class SpeedCommand:
    """Speed command output."""
    target_speed: float  # m/s
    acceleration: float  # m/s²
    brake_pressure: float  # 0-1
    reason: str  # Explanation for the command
    state: DrivingState  # Current driving state
    risk_level: str  # Risk level string
    action: str  # Action being taken


class SpeedSmoother:
    """Rule-based speed smoothing controller."""
    
    def __init__(
        self,
        max_speed: float = 20.0,  # m/s (~72 km/h)
        max_deceleration: float = -3.0,  # m/s² (comfortable braking)
        max_acceleration: float = 2.0,  # m/s²
        smoothing_factor: float = 0.1,  # How aggressive smoothing is (0-1)
        lookahead_time: float = 3.0  # seconds to look ahead
    ):
        """
        Initialize speed smoother.
        
        Args:
            max_speed: Maximum allowed speed in m/s
            max_deceleration: Maximum comfortable deceleration in m/s²
            max_acceleration: Maximum acceleration in m/s²
            smoothing_factor: Smoothing aggressiveness (0 = no smoothing, 1 = very smooth)
            lookahead_time: Time horizon for planning in seconds
        """
        self.max_speed = max_speed
        self.max_deceleration = max_deceleration
        self.max_acceleration = max_acceleration
        self.smoothing_factor = smoothing_factor
        self.lookahead_time = lookahead_time
        
        self.current_speed = 0.0
        self.target_speed_history: List[float] = []
        self.current_state = DrivingState.NORMAL
    
    def _calculate_safe_speed_for_obstacle(
        self, 
        obstacle: ObstacleInfo, 
        current_speed: float,
        dt: float = 0.1
    ) -> float:
        """
        Calculate safe speed given an obstacle.
        
        Args:
            obstacle: Obstacle information
            current_speed: Current vehicle speed
            dt: Time step
        
        Returns:
            Safe speed in m/s
        """
        # Base safe speed
        safe_speed = current_speed
        
        # If obstacle is approaching
        if obstacle.closing_trend == "approaching":
            # Calculate required deceleration
            # Higher relative speed = more aggressive braking needed
            relative_speed_factor = min(abs(obstacle.relative_speed) / 10.0, 1.0)
            
            # Distance factor (closer = slower)
            distance_factor = 1.0 - obstacle.distance_estimate
            
            # Calculate target speed reduction
            speed_reduction = current_speed * relative_speed_factor * distance_factor * 0.5
            safe_speed = max(0, current_speed - speed_reduction)
        
        # Traffic light logic
        if obstacle.is_traffic_light:
            if obstacle.traffic_light_state == "red":
                # Gradual slowdown for red light
                distance_to_light = obstacle.distance_estimate
                if distance_to_light < 0.3:  # Close to light
                    safe_speed = 0.0  # Stop
                else:
                    # Gradual deceleration
                    safe_speed = current_speed * distance_to_light
        
        return safe_speed
    
    def _apply_smoothing(
        self, 
        target_speed: float, 
        current_speed: float,
        dt: float = 0.1
    ) -> Tuple[float, float]:
        """
        Apply smoothing to speed changes to minimize acceleration spikes.
        
        Args:
            target_speed: Desired target speed
            current_speed: Current speed
            dt: Time step
        
        Returns:
            Tuple of (smoothed_target_speed, acceleration)
        """
        # Calculate required acceleration
        speed_diff = target_speed - current_speed
        
        # Apply smoothing factor
        smoothed_diff = speed_diff * self.smoothing_factor
        
        # Calculate acceleration (limited)
        acceleration = smoothed_diff / dt if dt > 0 else 0.0
        acceleration = np.clip(acceleration, self.max_deceleration, self.max_acceleration)
        
        # Calculate smoothed target speed
        smoothed_target = current_speed + acceleration * dt
        smoothed_target = np.clip(smoothed_target, 0, self.max_speed)
        
        return smoothed_target, acceleration
    
    def _determine_state(self, max_danger_score: float) -> DrivingState:
        """
        Determine driving state based on risk assessment.
        
        Args:
            max_danger_score: Maximum danger score from obstacles (0-1)
        
        Returns:
            DrivingState
        """
        if max_danger_score >= 0.7:
            return DrivingState.DANGER
        elif max_danger_score >= 0.3:
            return DrivingState.CAUTION
        else:
            return DrivingState.NORMAL
    
    def _get_state_behavior(self, state: DrivingState, current_speed: float, dt: float) -> Tuple[float, str]:
        """
        Get speed adjustment behavior based on state.
        
        Args:
            state: Current driving state
            current_speed: Current speed in m/s
            dt: Time step in seconds
        
        Returns:
            Tuple of (speed_change_per_second, action_description)
        """
        if state == DrivingState.NORMAL:
            # Maintain or slightly increase speed
            return 0.0, "MAINTAIN SPEED"
        elif state == DrivingState.CAUTION:
            # Gradually reduce speed: -1 km/h per second = -0.28 m/s per second
            speed_reduction_ms = -0.28  # -1 km/h per second
            return speed_reduction_ms, "SMOOTH DECELERATION"
        else:  # DANGER
            # Strong but smooth deceleration: -3 km/h per second = -0.83 m/s per second
            speed_reduction_ms = -0.83  # -3 km/h per second
            return speed_reduction_ms, "STRONG DECELERATION"
    
    def compute_speed_command(
        self,
        current_speed: float,
        obstacles: List[ObstacleInfo],
        gps_ok: bool,
        dt: float = 0.1,
        risk_assessments: Optional[List] = None
    ) -> SpeedCommand:
        """
        Compute speed command based on current state with risk-based state machine.
        
        Args:
            current_speed: Current vehicle speed in m/s
            obstacles: List of detected obstacles
            gps_ok: GPS reliability status
            dt: Time step in seconds
            risk_assessments: Optional list of risk assessments (from perception module)
        
        Returns:
            SpeedCommand with target speed, state, and control outputs
        """
        self.current_speed = current_speed
        
        # Find maximum danger score from risk assessments
        max_danger_score = 0.0
        max_risk_level = "LOW"
        
        if risk_assessments:
            for risk in risk_assessments:
                if hasattr(risk, 'danger_score'):
                    if risk.danger_score > max_danger_score:
                        max_danger_score = risk.danger_score
                        max_risk_level = risk.risk_level.value if hasattr(risk.risk_level, 'value') else str(risk.risk_level)
        
        # Determine driving state based on risk
        new_state = self._determine_state(max_danger_score)
        self.current_state = new_state
        
        # Get state-based behavior
        speed_change_per_sec, action = self._get_state_behavior(new_state, current_speed, dt)
        
        # Start with base target speed
        if gps_ok:
            if new_state == DrivingState.NORMAL:
                base_target_speed = min(self.max_speed, current_speed + 0.5)  # Can accelerate
            else:
                base_target_speed = current_speed  # Don't accelerate in CAUTION/DANGER
        else:
            # Conservative speed if GPS is unreliable
            base_target_speed = current_speed * 0.95  # Slight deceleration
            if new_state == DrivingState.NORMAL:
                new_state = DrivingState.CAUTION  # Elevate to CAUTION if GPS bad
        
        target_speed = base_target_speed
        
        # Apply state-based speed adjustment
        if new_state != DrivingState.NORMAL:
            # Apply gradual speed reduction based on state
            speed_adjustment = speed_change_per_sec * dt
            target_speed = max(0, current_speed + speed_adjustment)
            target_speed = min(target_speed, base_target_speed)  # Don't exceed base
        
        # Check obstacles for additional adjustments
        most_critical = None
        if obstacles:
            # Find most critical obstacle
            max_threat = 0.0
            
            for obstacle in obstacles:
                threat_score = 0.0
                
                if obstacle.closing_trend == "approaching":
                    threat_score += abs(obstacle.relative_speed) * (1.0 - obstacle.distance_estimate)
                
                if obstacle.is_traffic_light and obstacle.traffic_light_state == "red":
                    threat_score += 2.0 * (1.0 - obstacle.distance_estimate)
                
                if threat_score > max_threat:
                    max_threat = threat_score
                    most_critical = obstacle
            
            if most_critical:
                # Calculate safe speed for critical obstacle
                safe_speed = self._calculate_safe_speed_for_obstacle(
                    most_critical, current_speed, dt
                )
                target_speed = min(target_speed, safe_speed)
        
        # Apply smoothing
        smoothed_target, acceleration = self._apply_smoothing(target_speed, current_speed, dt)
        
        # Calculate brake pressure (0-1)
        if acceleration < 0:
            brake_pressure = min(abs(acceleration) / abs(self.max_deceleration), 1.0)
        else:
            brake_pressure = 0.0
        
        # Generate reason
        if not gps_ok:
            reason = "GPS unreliable - conservative speed"
        elif most_critical:
            if most_critical.is_traffic_light:
                reason = f"Traffic light {most_critical.traffic_light_state} detected"
            else:
                reason = f"Obstacle approaching - danger score: {max_danger_score:.2f}"
        else:
            reason = f"State: {new_state.value} - {action}"
        
        # Update history
        self.target_speed_history.append(smoothed_target)
        if len(self.target_speed_history) > 100:
            self.target_speed_history.pop(0)
        
        return SpeedCommand(
            target_speed=smoothed_target,
            acceleration=acceleration,
            brake_pressure=brake_pressure,
            reason=reason,
            state=new_state,
            risk_level=max_risk_level,
            action=action
        )
    
    def reset(self):
        """Reset smoother state."""
        self.current_speed = 0.0
        self.target_speed_history = []
        self.current_state = DrivingState.NORMAL