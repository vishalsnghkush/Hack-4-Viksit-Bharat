"""
Metrics Module
Computes speed variance, acceleration statistics, and brake events.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class SpeedMetrics:
    """Speed-related metrics."""
    mean_speed: float
    std_speed: float
    variance: float
    min_speed: float
    max_speed: float
    speed_range: float


@dataclass
class AccelerationMetrics:
    """Acceleration-related metrics."""
    mean_acceleration: float
    std_acceleration: float
    max_acceleration: float
    max_deceleration: float
    acceleration_peaks: int  # Number of peaks above threshold
    deceleration_peaks: int  # Number of deceleration peaks below threshold


@dataclass
class BrakeMetrics:
    """Braking-related metrics."""
    total_brake_events: int
    brake_event_rate: float  # Events per second
    total_brake_time: float  # Total time braking
    average_brake_pressure: float
    max_brake_pressure: float


@dataclass
class OverallMetrics:
    """Overall performance metrics."""
    speed_metrics: SpeedMetrics
    acceleration_metrics: AccelerationMetrics
    brake_metrics: BrakeMetrics
    total_time: float
    total_distance: float  # Estimated
    energy_efficiency_score: float  # 0-1, higher is better


class MetricsCollector:
    """Collects and computes metrics from speed data."""
    
    def __init__(
        self,
        acceleration_peak_threshold: float = 1.5,  # m/s²
        deceleration_peak_threshold: float = -1.5,  # m/s²
        brake_threshold: float = 0.1  # Brake pressure threshold
    ):
        """
        Initialize metrics collector.
        
        Args:
            acceleration_peak_threshold: Threshold for counting acceleration peaks
            deceleration_peak_threshold: Threshold for counting deceleration peaks
            brake_threshold: Minimum brake pressure to count as brake event
        """
        self.acceleration_peak_threshold = acceleration_peak_threshold
        self.deceleration_peak_threshold = deceleration_peak_threshold
        self.brake_threshold = brake_threshold
        
        # Data storage
        self.speed_history: deque = deque(maxlen=10000)
        self.acceleration_history: deque = deque(maxlen=10000)
        self.brake_pressure_history: deque = deque(maxlen=10000)
        self.timestamps: deque = deque(maxlen=10000)
        
        self.start_time: float = None
    
    def add_data_point(
        self,
        speed: float,
        acceleration: float,
        brake_pressure: float,
        timestamp: float = None
    ):
        """
        Add a data point.
        
        Args:
            speed: Current speed in m/s
            acceleration: Current acceleration in m/s²
            brake_pressure: Current brake pressure (0-1)
            timestamp: Timestamp (if None, uses internal counter)
        """
        import time
        if timestamp is None:
            timestamp = time.time()
        
        if self.start_time is None:
            self.start_time = timestamp
        
        self.speed_history.append(speed)
        self.acceleration_history.append(acceleration)
        self.brake_pressure_history.append(brake_pressure)
        self.timestamps.append(timestamp)
    
    def compute_speed_metrics(self) -> SpeedMetrics:
        """Compute speed-related metrics."""
        if len(self.speed_history) == 0:
            return SpeedMetrics(0, 0, 0, 0, 0, 0)
        
        speeds = np.array(self.speed_history)
        
        return SpeedMetrics(
            mean_speed=np.mean(speeds),
            std_speed=np.std(speeds),
            variance=np.var(speeds),
            min_speed=np.min(speeds),
            max_speed=np.max(speeds),
            speed_range=np.max(speeds) - np.min(speeds)
        )
    
    def compute_acceleration_metrics(self) -> AccelerationMetrics:
        """Compute acceleration-related metrics."""
        if len(self.acceleration_history) == 0:
            return AccelerationMetrics(0, 0, 0, 0, 0, 0)
        
        accelerations = np.array(self.acceleration_history)
        
        # Count peaks
        accel_peaks = np.sum(accelerations > self.acceleration_peak_threshold)
        decel_peaks = np.sum(accelerations < self.deceleration_peak_threshold)
        
        return AccelerationMetrics(
            mean_acceleration=np.mean(accelerations),
            std_acceleration=np.std(accelerations),
            max_acceleration=np.max(accelerations),
            max_deceleration=np.min(accelerations),
            acceleration_peaks=int(accel_peaks),
            deceleration_peaks=int(decel_peaks)
        )
    
    def compute_brake_metrics(self) -> BrakeMetrics:
        """Compute braking-related metrics."""
        if len(self.brake_pressure_history) == 0:
            return BrakeMetrics(0, 0, 0, 0, 0)
        
        brake_pressures = np.array(self.brake_pressure_history)
        timestamps = np.array(self.timestamps)
        
        # Count brake events (consecutive periods above threshold)
        brake_events = 0
        in_brake_event = False
        total_brake_time = 0.0
        brake_start_time = None
        
        for i, pressure in enumerate(brake_pressures):
            if pressure > self.brake_threshold:
                if not in_brake_event:
                    in_brake_event = True
                    brake_events += 1
                    brake_start_time = timestamps[i]
            else:
                if in_brake_event:
                    in_brake_event = False
                    if brake_start_time is not None:
                        total_brake_time += timestamps[i] - brake_start_time
        
        # Handle case where braking continues to end
        if in_brake_event and brake_start_time is not None:
            total_brake_time += timestamps[-1] - brake_start_time
        
        # Calculate brake event rate
        total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        brake_rate = brake_events / total_time if total_time > 0 else 0.0
        
        return BrakeMetrics(
            total_brake_events=brake_events,
            brake_event_rate=brake_rate,
            total_brake_time=total_brake_time,
            average_brake_pressure=np.mean(brake_pressures),
            max_brake_pressure=np.max(brake_pressures)
        )
    
    def compute_overall_metrics(self) -> OverallMetrics:
        """Compute overall performance metrics."""
        speed_metrics = self.compute_speed_metrics()
        acceleration_metrics = self.compute_acceleration_metrics()
        brake_metrics = self.compute_brake_metrics()
        
        # Calculate total time and distance
        if len(self.timestamps) > 1:
            total_time = self.timestamps[-1] - self.timestamps[0]
            # Estimate distance (integrate speed)
            speeds = np.array(self.speed_history)
            dt = total_time / len(speeds) if len(speeds) > 0 else 0.0
            total_distance = np.sum(speeds) * dt
        else:
            total_time = 0.0
            total_distance = 0.0
        
        # Calculate energy efficiency score (0-1, higher is better)
        # Lower variance, fewer brake events, smoother acceleration = better score
        speed_variance_score = 1.0 / (1.0 + speed_metrics.variance)  # Lower variance = higher score
        brake_score = 1.0 / (1.0 + brake_metrics.total_brake_events)  # Fewer brakes = higher score
        acceleration_smoothness = 1.0 / (1.0 + acceleration_metrics.std_acceleration)  # Smoother = higher score
        
        energy_efficiency_score = (speed_variance_score * 0.4 + brake_score * 0.4 + acceleration_smoothness * 0.2)
        
        return OverallMetrics(
            speed_metrics=speed_metrics,
            acceleration_metrics=acceleration_metrics,
            brake_metrics=brake_metrics,
            total_time=total_time,
            total_distance=total_distance,
            energy_efficiency_score=energy_efficiency_score
        )
    
    def get_comparison_report(self, baseline_metrics: OverallMetrics, smoothed_metrics: OverallMetrics) -> str:
        """
        Generate a comparison report between baseline and smoothed metrics.
        
        Args:
            baseline_metrics: Metrics without smoothing
            smoothed_metrics: Metrics with smoothing
        
        Returns:
            Formatted comparison report string
        """
        report = []
        report.append("=" * 60)
        report.append("METRICS COMPARISON REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Speed variance comparison
        var_improvement = ((baseline_metrics.speed_metrics.variance - smoothed_metrics.speed_metrics.variance) 
                          / baseline_metrics.speed_metrics.variance * 100) if baseline_metrics.speed_metrics.variance > 0 else 0
        report.append(f"Speed Variance:")
        report.append(f"  Baseline:  {baseline_metrics.speed_metrics.variance:.4f} m²/s²")
        report.append(f"  Smoothed:  {smoothed_metrics.speed_metrics.variance:.4f} m²/s²")
        report.append(f"  Improvement: {var_improvement:.1f}%")
        report.append("")
        
        # Brake events comparison
        brake_improvement = ((baseline_metrics.brake_metrics.total_brake_events - smoothed_metrics.brake_metrics.total_brake_events)
                            / baseline_metrics.brake_metrics.total_brake_events * 100) if baseline_metrics.brake_metrics.total_brake_events > 0 else 0
        report.append(f"Brake Events:")
        report.append(f"  Baseline:  {baseline_metrics.brake_metrics.total_brake_events}")
        report.append(f"  Smoothed:  {smoothed_metrics.brake_metrics.total_brake_events}")
        report.append(f"  Reduction: {brake_improvement:.1f}%")
        report.append("")
        
        # Acceleration peaks comparison
        accel_improvement = ((baseline_metrics.acceleration_metrics.acceleration_peaks - smoothed_metrics.acceleration_metrics.acceleration_peaks)
                            / baseline_metrics.acceleration_metrics.acceleration_peaks * 100) if baseline_metrics.acceleration_metrics.acceleration_peaks > 0 else 0
        report.append(f"Acceleration Peaks (> {self.acceleration_peak_threshold} m/s²):")
        report.append(f"  Baseline:  {baseline_metrics.acceleration_metrics.acceleration_peaks}")
        report.append(f"  Smoothed:  {smoothed_metrics.acceleration_metrics.acceleration_peaks}")
        report.append(f"  Reduction: {accel_improvement:.1f}%")
        report.append("")
        
        # Energy efficiency score
        eff_improvement = ((smoothed_metrics.energy_efficiency_score - baseline_metrics.energy_efficiency_score)
                           / baseline_metrics.energy_efficiency_score * 100) if baseline_metrics.energy_efficiency_score > 0 else 0
        report.append(f"Energy Efficiency Score:")
        report.append(f"  Baseline:  {baseline_metrics.energy_efficiency_score:.4f}")
        report.append(f"  Smoothed:  {smoothed_metrics.energy_efficiency_score:.4f}")
        report.append(f"  Improvement: {eff_improvement:.1f}%")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def reset(self):
        """Reset all collected data."""
        self.speed_history.clear()
        self.acceleration_history.clear()
        self.brake_pressure_history.clear()
        self.timestamps.clear()
        self.start_time = None
