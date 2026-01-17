"""
Demo Scenario Script
Creates before/after comparison runs showing the benefits of speed smoothing.
"""

import sys
from pathlib import Path
from main import VisionSpeedSmoothingSystem
from metrics import MetricsCollector
import matplotlib.pyplot as plt
import numpy as np


def run_demo_scenario(video_path: str, output_dir: str = "results", show_video: bool = True):
    """
    Run demo scenario with before/after comparison.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save results
        show_video: Whether to display video during processing
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*60)
    print("DEMO SCENARIO: Vision Speed Smoothing System")
    print("="*60)
    print("\nThis demo will run two scenarios:")
    print("1. Reactive mode (no smoothing) - baseline")
    print("2. Smoothing mode (with GPS degradation) - optimized")
    if not show_video:
        print("\nNote: Video display is disabled for faster processing.")
        print("      Remove --no-video flag to see the video during processing.")
    else:
        print("\nNote: You will see the video for BOTH runs.")
        print("      Press 'q' after the first run to continue to the second run.")
    print("\n" + "="*60 + "\n")
    
    # Run 1: Reactive (no smoothing, no GPS degradation)
    print("\n" + "="*60)
    print("RUN 1: REACTIVE MODE (Baseline)")
    print("="*60)
    system_reactive = VisionSpeedSmoothingSystem(
        video_path=video_path,
        enable_gps_degradation=False,
        enable_smoothing=False
    )
    metrics_reactive = system_reactive.run(display=show_video)
    
    # Save reactive metrics
    reactive_speeds = list(system_reactive.metrics.speed_history)
    reactive_accels = list(system_reactive.metrics.acceleration_history)
    reactive_brakes = list(system_reactive.metrics.brake_pressure_history)
    
    # Run 2: Smoothing (with smoothing and GPS degradation)
    print("\n" + "="*60)
    print("RUN 2: SMOOTHING MODE (Optimized)")
    print("="*60)
    system_smooth = VisionSpeedSmoothingSystem(
        video_path=video_path,
        enable_gps_degradation=True,
        enable_smoothing=True
    )
    metrics_smooth = system_smooth.run(display=show_video)
    
    # Save smooth metrics
    smooth_speeds = list(system_smooth.metrics.speed_history)
    smooth_accels = list(system_smooth.metrics.acceleration_history)
    smooth_brakes = list(system_smooth.metrics.brake_pressure_history)
    
    # Generate comparison report
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)
    comparison = system_smooth.metrics.get_comparison_report(metrics_reactive, metrics_smooth)
    print(comparison)
    
    # Save report to file
    report_path = output_path / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(comparison)
    print(f"\nReport saved to: {report_path}")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    
    # Create time axis
    time_reactive = np.linspace(0, metrics_reactive.total_time, len(reactive_speeds))
    time_smooth = np.linspace(0, metrics_smooth.total_time, len(smooth_speeds))
    
    # Plot 1: Speed comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(time_reactive, reactive_speeds, label='Reactive', alpha=0.7, linewidth=1)
    plt.plot(time_smooth, smooth_speeds, label='Smoothed', alpha=0.7, linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Acceleration comparison
    plt.subplot(2, 2, 2)
    plt.plot(time_reactive, reactive_accels, label='Reactive', alpha=0.7, linewidth=1)
    plt.plot(time_smooth, smooth_accels, label='Smoothed', alpha=0.7, linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Acceleration Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Brake pressure comparison
    plt.subplot(2, 2, 3)
    plt.plot(time_reactive, reactive_brakes, label='Reactive', alpha=0.7, linewidth=1)
    plt.plot(time_smooth, smooth_brakes, label='Smoothed', alpha=0.7, linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Brake Pressure')
    plt.title('Brake Pressure Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Metrics comparison
    plt.subplot(2, 2, 4)
    metrics_names = ['Speed\nVariance', 'Brake\nEvents', 'Accel\nPeaks', 'Energy\nEfficiency']
    reactive_values = [
        metrics_reactive.speed_metrics.variance,
        metrics_reactive.brake_metrics.total_brake_events,
        metrics_reactive.acceleration_metrics.acceleration_peaks,
        metrics_reactive.energy_efficiency_score
    ]
    smooth_values = [
        metrics_smooth.speed_metrics.variance,
        metrics_smooth.brake_metrics.total_brake_events,
        metrics_smooth.acceleration_metrics.acceleration_peaks,
        metrics_smooth.energy_efficiency_score
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    # Normalize for comparison (except efficiency score)
    reactive_norm = [v / max(reactive_values[i], 1) for i, v in enumerate(reactive_values)]
    smooth_norm = [v / max(reactive_values[i], 1) for i, v in enumerate(smooth_values)]
    
    plt.bar(x - width/2, reactive_norm, width, label='Reactive', alpha=0.7)
    plt.bar(x + width/2, smooth_norm, width, label='Smoothed', alpha=0.7)
    plt.xlabel('Metrics')
    plt.ylabel('Normalized Value')
    plt.title('Metrics Comparison (Normalized)')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_path / "comparison_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_path}")
    print("\nKey Improvements:")
    var_improvement = ((metrics_reactive.speed_metrics.variance - metrics_smooth.speed_metrics.variance) 
                      / metrics_reactive.speed_metrics.variance * 100) if metrics_reactive.speed_metrics.variance > 0 else 0
    brake_improvement = ((metrics_reactive.brake_metrics.total_brake_events - metrics_smooth.brake_metrics.total_brake_events)
                        / metrics_reactive.brake_metrics.total_brake_events * 100) if metrics_reactive.brake_metrics.total_brake_events > 0 else 0
    print(f"  - Speed variance reduced by {var_improvement:.1f}%")
    print(f"  - Brake events reduced by {brake_improvement:.1f}%")
    print(f"  - Energy efficiency improved by {((metrics_smooth.energy_efficiency_score - metrics_reactive.energy_efficiency_score) / metrics_reactive.energy_efficiency_score * 100):.1f}%")


def main():
    """Main entry point for demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo scenario for vision speed smoothing")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    parser.add_argument("--no-video", action="store_true", help="Disable video display during processing")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Default video
    if args.video:
        video_path = args.video
        if not Path(video_path).is_absolute():
            video_path = project_root / video_path
    else:
        video_path = data_dir / "8359-208052066_small.mp4"
    
    run_demo_scenario(str(video_path), args.output, show_video=not args.no_video)


if __name__ == "__main__":
    main()
