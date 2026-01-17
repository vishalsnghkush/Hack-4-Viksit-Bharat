"""
GPS Monitor Module
Monitors GPS reliability and detects jitter, signal loss, and other degradation modes.
"""

import time
import random
import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque
from dataclasses import dataclass
from enum import Enum


class GPSStatus(Enum):
    """GPS status states."""
    OK = "OK"
    HIGH_JITTER = "HIGH_JITTER"
    FROZEN = "FROZEN"
    DROPOUT = "DROPOUT"
    JUMP = "JUMP"


@dataclass
class GPSReading:
    """GPS reading data structure."""
    latitude: float
    longitude: float
    timestamp: float
    speed: float  # m/s
    heading: float  # degrees


class GPSDegradationSimulator:
    """Simulates GPS degradation scenarios."""
    
    def __init__(
        self,
        base_lat: float = 37.7749,
        base_lon: float = -122.4194,
        base_speed: float = 15.0,  # m/s (~54 km/h)
        degradation_prob: float = 0.02  # Reduced from 0.1 to 0.02 (2% chance per update)
    ):
        """
        Initialize GPS degradation simulator.
        
        Args:
            base_lat: Base latitude
            base_lon: Base longitude
            base_speed: Base speed in m/s
            degradation_prob: Probability of degradation event per update
        """
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.base_speed = base_speed
        self.degradation_prob = degradation_prob
        
        self.current_lat = base_lat
        self.current_lon = base_lon
        self.current_speed = base_speed
        self.current_heading = 0.0
        
        self.frozen = False
        self.frozen_value = None
        self.dropout = False
        self.jump_countdown = 0
        
        self.time_elapsed = 0.0
        self.last_update_time = time.time()
    
    def generate_normal_reading(self, dt: float) -> GPSReading:
        """Generate a normal GPS reading with small noise."""
        # Simulate movement
        speed_ms = self.base_speed
        heading_rad = np.radians(self.current_heading)
        
        # Update position (simplified - assumes small movements)
        lat_change = (speed_ms * dt * np.cos(heading_rad)) / 111320.0  # meters to degrees
        lon_change = (speed_ms * dt * np.sin(heading_rad)) / (111320.0 * np.cos(np.radians(self.current_lat)))
        
        self.current_lat += lat_change + random.gauss(0, 0.00001)  # Small noise
        self.current_lon += lon_change + random.gauss(0, 0.00001)
        
        # Add small speed variation
        self.current_speed = speed_ms + random.gauss(0, 0.5)
        
        return GPSReading(
            latitude=self.current_lat,
            longitude=self.current_lon,
            timestamp=time.time(),
            speed=self.current_speed,
            heading=self.current_heading
        )
    
    def generate_degraded_reading(self, dt: float) -> Optional[GPSReading]:
        """Generate a degraded GPS reading based on failure mode."""
        if self.dropout:
            # Dropout: return None (no GPS signal)
            if random.random() < 0.1:  # 10% chance to recover
                self.dropout = False
            return None
        
        if self.frozen:
            # Frozen: return last known value
            if random.random() < 0.05:  # 5% chance to unfreeze
                self.frozen = False
            return self.frozen_value
        
        if self.jump_countdown > 0:
            # Jump: sudden position change
            self.current_lat += random.gauss(0, 0.001)  # Large jump
            self.current_lon += random.gauss(0, 0.001)
            self.jump_countdown -= 1
            return GPSReading(
                latitude=self.current_lat,
                longitude=self.current_lon,
                timestamp=time.time(),
                speed=self.current_speed,
                heading=self.current_heading
            )
        
        # Normal reading with high jitter
        reading = self.generate_normal_reading(dt)
        reading.latitude += random.gauss(0, 0.0001)  # High jitter
        reading.longitude += random.gauss(0, 0.0001)
        return reading
    
    def update(self, dt: float) -> Optional[GPSReading]:
        """
        Generate next GPS reading with potential degradation.
        
        Args:
            dt: Time delta since last update
        
        Returns:
            GPSReading or None (if dropout)
        """
        self.time_elapsed += dt
        
        # Check for new degradation event
        if random.random() < self.degradation_prob:
            degradation_type = random.choice(['jitter', 'frozen', 'dropout', 'jump'])
            
            if degradation_type == 'frozen':
                self.frozen = True
                self.frozen_value = GPSReading(
                    latitude=self.current_lat,
                    longitude=self.current_lon,
                    timestamp=time.time(),
                    speed=self.current_speed,
                    heading=self.current_heading
                )
            elif degradation_type == 'dropout':
                self.dropout = True
            elif degradation_type == 'jump':
                self.jump_countdown = random.randint(3, 10)
        
        # Generate reading
        if self.dropout or self.frozen or self.jump_countdown > 0:
            return self.generate_degraded_reading(dt)
        else:
            return self.generate_normal_reading(dt)


class GPSMonitor:
    """Monitors GPS reliability and flags issues."""
    
    def __init__(
        self,
        jitter_threshold: float = 0.0001,  # degrees
        max_no_update_time: float = 2.0,  # seconds
        jump_threshold: float = 0.001  # degrees
    ):
        """
        Initialize GPS monitor.
        
        Args:
            jitter_threshold: Maximum acceptable position variation (degrees)
            max_no_update_time: Maximum time without update before flagging dropout
            jump_threshold: Threshold for detecting position jumps (degrees)
        """
        self.jitter_threshold = jitter_threshold
        self.max_no_update_time = max_no_update_time
        self.jump_threshold = jump_threshold
        
        self.reading_history: deque = deque(maxlen=20)
        self.last_reading: Optional[GPSReading] = None
        self.last_update_time: Optional[float] = None
        
        self.gps_ok = True
        self.status = GPSStatus.OK
        self.status_message = "GPS OK"
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in degrees."""
        return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    
    def _detect_jitter(self) -> bool:
        """Detect high jitter in recent readings."""
        if len(self.reading_history) < 5:
            return False
        
        positions = [(r.latitude, r.longitude) for r in self.reading_history]
        distances = []
        
        for i in range(1, len(positions)):
            dist = self._calculate_distance(
                positions[i-1][0], positions[i-1][1],
                positions[i][0], positions[i][1]
            )
            distances.append(dist)
        
        if len(distances) == 0:
            return False
        
        # High jitter if variance is high
        avg_distance = np.mean(distances)
        return avg_distance > self.jitter_threshold
    
    def _detect_frozen(self) -> bool:
        """Detect if GPS values are frozen."""
        if len(self.reading_history) < 3:
            return False
        
        positions = [(r.latitude, r.longitude) for r in self.reading_history]
        
        # Check if all recent positions are identical (within small tolerance)
        first_pos = positions[0]
        for pos in positions[1:]:
            dist = self._calculate_distance(first_pos[0], first_pos[1], pos[0], pos[1])
            if dist > 0.00001:  # Very small threshold
                return False
        
        return True
    
    def _detect_jump(self) -> bool:
        """Detect sudden position jumps."""
        if len(self.reading_history) < 2:
            return False
        
        if self.last_reading is None:
            return False
        
        current = self.reading_history[-1]
        dist = self._calculate_distance(
            self.last_reading.latitude, self.last_reading.longitude,
            current.latitude, current.longitude
        )
        
        return dist > self.jump_threshold
    
    def _detect_dropout(self) -> bool:
        """Detect GPS signal dropout."""
        if self.last_update_time is None:
            return False
        
        time_since_update = time.time() - self.last_update_time
        return time_since_update > self.max_no_update_time
    
    def update(self, reading: Optional[GPSReading]) -> Dict:
        """
        Update GPS monitor with new reading.
        
        Args:
            reading: GPSReading or None (if dropout)
        
        Returns:
            Dictionary with GPS status information
        """
        current_time = time.time()
        
        if reading is None:
            # Dropout detected
            self.gps_ok = False
            self.status = GPSStatus.DROPOUT
            self.status_message = "GPS DROPOUT - No signal"
        else:
            # Add to history
            self.reading_history.append(reading)
            self.last_update_time = current_time
            
            # Check for various issues
            if self._detect_jump():
                self.gps_ok = False
                self.status = GPSStatus.JUMP
                self.status_message = f"GPS JUMP detected - Position changed by {self._calculate_distance(self.last_reading.latitude, self.last_reading.longitude, reading.latitude, reading.longitude):.6f} degrees"
            elif self._detect_frozen():
                self.gps_ok = False
                self.status = GPSStatus.FROZEN
                self.status_message = "GPS FROZEN - No position updates"
            elif self._detect_jitter():
                self.gps_ok = False
                self.status = GPSStatus.HIGH_JITTER
                self.status_message = "GPS HIGH JITTER - Unstable readings"
            else:
                self.gps_ok = True
                self.status = GPSStatus.OK
                self.status_message = "GPS OK"
            
            self.last_reading = reading
        
        # Also check for dropout based on time
        if self._detect_dropout():
            self.gps_ok = False
            self.status = GPSStatus.DROPOUT
            self.status_message = "GPS DROPOUT - No update for too long"
        
        return {
            'gps_ok': self.gps_ok,
            'status': self.status.value,
            'message': self.status_message,
            'reading': reading
        }
    
    def get_status(self) -> Dict:
        """
        Get current GPS status.
        
        Returns:
            Dictionary with current GPS status
        """
        return {
            'gps_ok': self.gps_ok,
            'status': self.status.value,
            'message': self.status_message,
            'last_reading': self.last_reading
        }
