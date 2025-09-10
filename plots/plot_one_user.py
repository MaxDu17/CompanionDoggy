# write a script to process gpx files and be able to get the instantaneous speed and plot that

import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
from datetime import datetime
import math
import os
import numpy as np

class GPXProcessor:
    def __init__(self, file_path):
        self.gpx_data = self.load_gpx_file(file_path)
        self.instantaneous_speed = []
        self.per5sec_speeds = []
        self.time_diffs = []
        
    def load_gpx_file(self, file_path):
        gpx_file = open(file_path, 'r')
        self.gpx_data = gpxpy.parse(gpx_file)
        return self.gpx_data

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371000  # Earth's radius in meters
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def get_instantaneous_speed(self, time_interval=None):
        # get the instantaneous speed
        instantaneous_speed = []
        time_diffs = []
        
        for track in self.gpx_data.tracks:
            for segment in track.segments:
                points = segment.points
                
                if time_interval is None:
                    # Original behavior - no aggregation
                    for i in range(len(points)-1):
                        point1 = points[i]
                        point2 = points[i+1]
                        
                        # Calculate distance between consecutive points
                        distance = self.calculate_distance(
                            point1.latitude, point1.longitude,
                            point2.latitude, point2.longitude
                        )
                        
                        # Calculate time difference in seconds
                        time1 = point1.time
                        time2 = point2.time
                        if time1 and time2:  # Check if timestamps exist
                            time_diff = (time2 - time1).total_seconds()
                            time_diffs.append(time_diff)
                            if time_diff > 0:  # Avoid division by zero
                                # Speed in meters per second
                                speed = distance / time_diff
                                instantaneous_speed.append(speed)
                else:
                    # Aggregate until time threshold is met
                    cumulative_distance = 0
                    cumulative_time = 0
                    start_time = None
                    
                    for i in range(len(points)-1):
                        point1 = points[i]
                        point2 = points[i+1]
                        
                        # Calculate distance between consecutive points
                        distance = self.calculate_distance(
                            point1.latitude, point1.longitude,
                            point2.latitude, point2.longitude
                        )
                        
                        # Calculate time difference in seconds
                        time1 = point1.time
                        time2 = point2.time
                        
                        if time1 and time2:  # Check if timestamps exist
                            time_diff = (time2 - time1).total_seconds()
                            
                            if start_time is None:
                                start_time = time1
                            
                            cumulative_distance += distance
                            cumulative_time += time_diff
                            
                            # If we've reached the time threshold, calculate speed
                            if cumulative_time >= time_interval:
                                if cumulative_time > 0:  # Avoid division by zero
                                    # Speed in meters per second
                                    speed = cumulative_distance / cumulative_time
                                    instantaneous_speed.append(speed)
                                    time_diffs.append(cumulative_time)
                                
                                # Reset for next aggregation
                                cumulative_distance = 0
                                cumulative_time = 0
                                start_time = None
                    
                    # Handle any remaining aggregated data
                    if cumulative_time > 0 and cumulative_time >= time_interval:
                        speed = cumulative_distance / cumulative_time
                        instantaneous_speed.append(speed)
                        time_diffs.append(cumulative_time)
        
        return instantaneous_speed, time_diffs

    def get_timed_speeds(self, time_interval=5, plot=False, use_mph=False):
        instantaneous_speed, time_diffs = self.get_instantaneous_speed(time_interval)
        timed_speeds = []
        current_window_speeds = []
        current_window_time = 0
        
        for i in range(len(instantaneous_speed)):
            current_window_speeds.append(instantaneous_speed[i])
            current_window_time += time_diffs[i]
            
            if current_window_time >= time_interval:
                # Calculate average speed over the time window
                if current_window_speeds:
                    avg_speed = np.mean(current_window_speeds)
                    timed_speeds.append(avg_speed)
                
                # Reset for next window
                current_window_speeds = []
                current_window_time = 0
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(timed_speeds, 'b-', linewidth=2)
            plt.xlabel('Time Window Index')
            
            # Calculate area under the m/s curve (total distance)
            # Area = sum of speeds * time_interval (since each point represents average speed over time_interval)
            total_distance = sum(timed_speeds) * time_interval
            print(f"Area under m/s curve (total distance): {total_distance:.2f} meters ({total_distance/1000:.2f} km)")
            
            if use_mph:
                # Convert m/s to mph (1 m/s = 2.23694 mph)
                mph_speeds = [speed * 2.23694 for speed in timed_speeds]
                plt.plot(mph_speeds, 'r-', linewidth=2)
                plt.ylabel('Average Speed (mph)')
                plt.title(f'Average Speed Over {time_interval}-Second Windows (mph)')
                plt.legend(['m/s', 'mph'])
            else:
                plt.ylabel('Average Speed (m/s)')
                plt.title(f'Average Speed Over {time_interval}-Second Windows')
            
            plt.grid(True, alpha=0.3)
            plt.show()
        return timed_speeds

    def get_xy_positions(self, plot=False):
        x_positions = []
        y_positions = []
        for track in self.gpx_data.tracks:
            for segment in track.segments:
                points = segment.points
                for point in points:
                    x_positions.append(point.longitude)
                    y_positions.append(point.latitude)
        if plot:
            plt.figure(figsize=(10, 8))
            plt.scatter(x_positions, y_positions, alpha=0.6, s=20)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('GPS Track')
            plt.grid(True, alpha=0.3)
            plt.show()
        return x_positions, y_positions

    def analyze_interval_training(self, fast_target, slow_target, start_timestamp=None, chunk_duration=30, total_duration=120, plot=True):
        """
        Analyze interval training performance with alternating fast/slow targets.
        
        Args:
            fast_target: Target speed for fast intervals (m/s)
            slow_target: Target speed for slow intervals (m/s)
            start_timestamp: Timestamp to start analysis from (if None, finds first fast target reach)
            chunk_duration: Duration of each interval chunk in seconds (default 30)
            total_duration: Total duration to analyze in seconds (default 120 = 2 minutes)
            plot: Whether to create a plot showing the analysis (default True)
        """
        # Get smoothed speeds using get_timed_speeds for better analysis
        # First, get all the timed speeds with small intervals for good resolution
        all_timed_speeds = self.get_timed_speeds(time_interval=5, plot=False)  # 5-second smoothed speeds
        
        if not all_timed_speeds:
            print("No speed data available")
            return
        
        # Calculate the start index in the timed speeds array
        start_index = None
        
        if start_timestamp is not None:
            # Convert timestamp to index in timed_speeds (each point represents 5 seconds)
            start_index = int(start_timestamp // 5)  # 5-second intervals
            if start_index >= len(all_timed_speeds):
                print(f"Timestamp {start_timestamp} seconds exceeds available data")
                return
            print(f"Starting analysis at timestamp {start_timestamp} seconds (timed speed index {start_index})")
        else:
            # Find when speed first reaches fast target (original behavior)
            for i, speed in enumerate(all_timed_speeds):
                if speed >= fast_target:
                    start_index = i
                    break
            
            if start_index is None:
                print(f"Speed never reached fast target of {fast_target:.2f} m/s")
                return
            print(f"Fast target speed ({fast_target:.2f} m/s) reached at timed speed index {start_index}")
        
        # Extract the required duration of data (e.g., 2 minutes = 120 seconds)
        # With 5-second intervals, we need 120/5 = 24 data points
        points_needed = int(total_duration // 5)
        end_index = min(start_index + points_needed, len(all_timed_speeds))
        
        analysis_speeds = all_timed_speeds[start_index:end_index]
        
        if len(analysis_speeds) < points_needed:
            actual_duration = len(analysis_speeds) * 5
            print(f"Warning: Only {actual_duration} seconds of data available after start point (needed {total_duration})")
        
        print(f"Using {len(analysis_speeds)} smoothed speed points (5-second intervals) for analysis")
        
        # Segment into 30-second chunks using smoothed speeds
        # With 5-second intervals, each chunk should have 30/5 = 6 speed points
        points_per_chunk = int(chunk_duration // 5)
        chunks = []
        
        # For plotting - track cumulative time and analysis regions
        plot_data = {
            'times': [],
            'speeds': [],
            'chunk_boundaries': [],
            'analysis_regions': [],
            'targets': []
        }
        
        # Create time array for plotting (each point represents 5 seconds)
        for i, speed in enumerate(analysis_speeds):
            plot_data['times'].append(i * 5)  # 5-second intervals
            plot_data['speeds'].append(speed)
        
        chunk_index = 0
        for chunk_start_idx in range(0, len(analysis_speeds), points_per_chunk):
            chunk_end_idx = min(chunk_start_idx + points_per_chunk, len(analysis_speeds))
            
            if chunk_end_idx - chunk_start_idx < points_per_chunk:
                # Skip incomplete chunks at the end
                break
                
            current_chunk_speeds = analysis_speeds[chunk_start_idx:chunk_end_idx]
            
            if len(current_chunk_speeds) >= points_per_chunk:
                # Determine if this should be fast or slow (alternating, starting with fast)
                is_fast_chunk = (chunk_index % 2 == 0)
                target_speed = fast_target if is_fast_chunk else slow_target
                chunk_type = "Fast" if is_fast_chunk else "Slow"
                
                # Store chunk boundary and target for plotting
                chunk_start_time = chunk_start_idx * 5
                chunk_end_time = chunk_end_idx * 5
                plot_data['chunk_boundaries'].append((chunk_start_time, chunk_end_time))
                plot_data['targets'].append((chunk_start_time, chunk_end_time, target_speed, chunk_type))
                
                # Store analysis region (middle 20 seconds) for plotting
                # With 5-second intervals: skip first point (0-5s) and last point (25-30s), use middle 4 points (5-25s)
                analysis_start_time = chunk_start_time + 5
                analysis_end_time = chunk_end_time - 5
                plot_data['analysis_regions'].append((analysis_start_time, analysis_end_time, target_speed, chunk_type))
                
                # Extract middle 4 points (middle 20 seconds: skip first and last point of the 6-point chunk)
                if len(current_chunk_speeds) >= 4:
                    middle_speeds = current_chunk_speeds[1:-1]  # Skip first and last point
                    analysis_note = f"(middle 20s: {len(middle_speeds)} of {len(current_chunk_speeds)} smoothed points)"
                else:
                    # Fallback to full chunk if not enough points
                    middle_speeds = current_chunk_speeds
                    analysis_note = "(full chunk - not enough points for middle extraction)"
                
                # Calculate statistics for the middle portion
                mean_speed = np.mean(middle_speeds)
                variance = np.var(middle_speeds)
                speed_diff = mean_speed - target_speed
                
                chunk_data = {
                    'index': chunk_index,
                    'type': chunk_type,
                    'target': target_speed,
                    'mean_speed': mean_speed,
                    'variance': variance,
                    'speed_diff': speed_diff,
                    'duration': chunk_duration,  # Always 30 seconds for complete chunks
                    'analysis_note': analysis_note
                }
                chunks.append(chunk_data)
                chunk_index += 1
        
        # Report results
        print(f"\n=== Interval Training Analysis ===")
        print(f"Fast target: {fast_target:.2f} m/s ({fast_target * 2.23694:.2f} mph)")
        print(f"Slow target: {slow_target:.2f} m/s ({slow_target * 2.23694:.2f} mph)")
        print(f"Chunk duration: {chunk_duration} seconds (analyzing middle 20 seconds of each)")
        print(f"Total analyzed duration: {sum(chunk['duration'] for chunk in chunks):.1f} seconds")
        print(f"\nNumber of chunks analyzed: {len(chunks)}")
        
        print(f"\n{'Chunk':<5} {'Type':<4} {'Target':<8} {'Mean':<8} {'Diff':<8} {'Variance':<10} {'Note':<25}")
        print("-" * 75)
        
        total_speed_diff = 0
        total_variance = 0
        
        for chunk in chunks:
            print(f"{chunk['index']:<5} {chunk['type']:<4} {chunk['target']:<8.2f} "
                  f"{chunk['mean_speed']:<8.2f} {chunk['speed_diff']:<8.2f} {chunk['variance']:<10.4f} {chunk['analysis_note']:<25}")
            total_speed_diff += abs(chunk['speed_diff'])
            total_variance += chunk['variance']
        
        # Calculate aggregates
        if chunks:
            avg_speed_diff = total_speed_diff / len(chunks)
            avg_variance = total_variance / len(chunks)
            
            print(f"\n=== Aggregate Statistics ===")
            print(f"Average absolute difference from target per chunk: {avg_speed_diff:.4f} m/s")
            print(f"Average variance per chunk: {avg_variance:.4f} (m/s)²")
            
            # Convert to mph for readability
            print(f"Average absolute difference from target per chunk: {avg_speed_diff * 2.23694:.4f} mph")
            print(f"Average standard deviation per chunk: {np.sqrt(avg_variance) * 2.23694:.4f} mph")
        
        # Create the plot if requested
        if plot and plot_data['times']:
            self._plot_interval_analysis(plot_data, fast_target, slow_target, chunk_duration)
        
        return chunks

    def _extract_middle_speeds(self, chunk_speeds, chunk_times, skip_start_secs, skip_end_secs):
        """
        Extract speeds from the middle portion of a chunk, skipping the specified
        seconds at the beginning and end.
        """
        if not chunk_speeds or not chunk_times:
            return []
        
        # Find the start and end indices for the middle portion
        cumulative_time = 0
        start_idx = None
        end_idx = None
        
        # Find start index (after skip_start_secs)
        for i, time_diff in enumerate(chunk_times):
            cumulative_time += time_diff
            if cumulative_time >= skip_start_secs and start_idx is None:
                start_idx = i
            if cumulative_time >= (sum(chunk_times) - skip_end_secs) and end_idx is None:
                end_idx = i
                break
        
        if start_idx is None or end_idx is None or start_idx >= end_idx:
            return []
        
        return chunk_speeds[start_idx:end_idx+1]

    def _plot_interval_analysis(self, plot_data, fast_target, slow_target, chunk_duration):
        """
        Create a plot showing the speed data with interval analysis regions highlighted.
        """
        plt.figure(figsize=(8, 8))
        
        # Plot the speed data
        plt.plot(plot_data['times'], plot_data['speeds'], 'b-', linewidth=1.5, alpha=0.7, label='Speed data')
        
        # Plot target lines for each chunk
        for start_time, end_time, target_speed, chunk_type in plot_data['targets']:
            color = 'red' if chunk_type == 'Fast' else 'green'
            plt.axhline(y=target_speed, xmin=start_time/max(plot_data['times']), 
                       xmax=end_time/max(plot_data['times']), color=color, 
                       linewidth=3, alpha=0.6, 
                       label=f'{chunk_type} target ({target_speed:.1f} m/s)' if start_time == min([t[0] for t in plot_data['targets'] if t[3] == chunk_type]) else "")
        
        # Highlight chunk boundaries
        for i, (start_time, end_time) in enumerate(plot_data['chunk_boundaries']):
            plt.axvspan(start_time, end_time, alpha=0.1, color='gray', 
                       label='30s chunks' if i == 0 else "")
        
        # Highlight analysis regions (middle 20 seconds)
        for i, (start_time, end_time, target_speed, chunk_type) in enumerate(plot_data['analysis_regions']):
            color = 'red' if chunk_type == 'Fast' else 'green'
            plt.axvspan(start_time, end_time, alpha=0.2, color=color,
                       label='Analysis regions (middle 20s)' if i == 0 else "")
        
        # Add vertical lines for chunk boundaries
        for start_time, end_time in plot_data['chunk_boundaries']:
            plt.axvline(x=start_time, color='black', linestyle='--', alpha=0.5, linewidth=1)
            plt.axvline(x=end_time, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add vertical lines for analysis region boundaries
        for start_time, end_time, _, _ in plot_data['analysis_regions']:
            plt.axvline(x=start_time, color='orange', linestyle=':', alpha=0.7, linewidth=2)
            plt.axvline(x=end_time, color='orange', linestyle=':', alpha=0.7, linewidth=2)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Speed (m/s)')
        plt.title('Interval Training Analysis\n(Orange dotted lines show analyzed regions)')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

def plot_user4_priya_speeds():
    """
    Plot User 4 Priya's speed data for all three conditions in a style similar to process_prims.py
    """
    # User 4 Priya start times (from user_start_times.txt)
    start_times = {
        "control": 108,  # seconds
        "watch": 38,     # seconds  
        "dog": 68        # seconds
    }
    
    # file_dir = "/Users/jennifergrannen/Documents/Stanford/iliad/CompanionDoggy/user_study_data/user4_Priya/"
    file_dir = "/Users/maxjdu/Documents/GitHub/CompanionDoggy/logs/4_Priya"
    files = ["control.gpx", "watch.gpx", "dog.gpx"]
    method_key = {"control": "No Pacer", "watch": "Watch", "dog": "Dog"}
    colors = {"control": "darkgrey", "watch": "lightblue", "dog": "mediumseagreen"}
    
    # Store speed data for each condition
    all_speed_data = {}
    
    for file in files:
        condition = file.replace(".gpx", "")
        filepath = os.path.join(file_dir, file)
        
        print(f"\nProcessing {condition} data...")
        gpx_processor = GPXProcessor(filepath)
        
        # Get timed speeds (5-second intervals)
        timed_speeds = gpx_processor.get_timed_speeds(time_interval=5, plot=False, use_mph=False)
        
        # Extract data starting from the specified start time
        start_index = start_times[condition] // 5  # Convert seconds to 5-second intervals
        if start_index < len(timed_speeds):
            # Take 80 seconds of data (16 data points at 5-second intervals)
            end_index = min(start_index + 16, len(timed_speeds))
            condition_speeds = timed_speeds[start_index:end_index]
            all_speed_data[condition] = condition_speeds
            print(f"  {condition}: {len(condition_speeds)} data points from {start_times[condition]}s (80 seconds)")
        else:
            print(f"  {condition}: Start time {start_times[condition]}s exceeds available data")
            all_speed_data[condition] = []
    
    # Create the plot similar to process_prims.py
    # fig, ax = plt.subplots(figsize=(6, 5))
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Time points (5-second intervals)
    time_points = np.arange(0, 80, 5)  # 0, 5, 10, ..., 75 seconds
    
    # Plot each condition
    for condition in ["control", "watch", "dog"]:
        if all_speed_data[condition]:
            speeds = all_speed_data[condition]
            # Pad with NaN if we don't have enough data points
            if len(speeds) < 16:
                speeds = speeds + [np.nan] * (16 - len(speeds))
            
            ax.plot(time_points, speeds, 'o-', color=colors[condition], 
                   linewidth=2, label=method_key[condition], markersize=4)
    
    # Styling similar to process_prims.py
    # ax.set_xlabel('Time (seconds)', fontsize=12, font="Palatino")
    # ax.set_ylabel('Speed (m/s)', fontsize=12, font="Palatino")
    # ax.set_title('User 4 Priya - Speed Over Time', fontsize=14, font="Palatino", pad=20)
    
    # Set x-axis ticks every 10 seconds but only label the 30-second intervals
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
    ax.set_xticklabels(['0', '', '', '30', '', '', '60', '', '80'], font="Palatino")
    
    # Set y-axis ticks
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels([0, 1, 2, 3, 4, 5], font="Palatino")
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    # ax.legend(frameon=False, loc='upper right', fontfamily="Palatino")
    
    # Add alternating target speed line (fast-slow-fast pattern every 30 seconds)
    fast_target = 2.8  # m/s
    slow_target = 1.8  # m/s
    
    # Create time and target arrays for the alternating pattern
    target_times = []
    target_speeds = []
    target_colors = []
    
    # 0-30s: Fast (red)
    target_times.extend([0, 30])
    target_speeds.extend([fast_target, fast_target])
    target_colors.extend(['red', 'red'])
    
    # 30-60s: Slow (green) 
    target_times.extend([30, 60])
    target_speeds.extend([slow_target, slow_target])
    target_colors.extend(['green', 'green'])
    
    # 60-80s: Fast (red)
    target_times.extend([60, 80])
    target_speeds.extend([fast_target, fast_target])
    target_colors.extend(['red', 'red'])
    
    # Plot the alternating target line
    ax.plot(target_times, target_speeds, '--', color='red', alpha=0.7, linewidth=2, 
           label='Target Pattern')
    
    # # Add colored segments for visual distinction (all red)
    # ax.plot([0, 30], [fast_target, fast_target], '--', color='red', alpha=0.7, linewidth=2)
    # ax.plot([30, 60], [slow_target, slow_target], '--', color='red', alpha=0.7, linewidth=2)
    # ax.plot([60, 80], [fast_target, fast_target], '--', color='red', alpha=0.7, linewidth=2)
    
    # # Add grid
    # ax.grid(True, alpha=0.3)
    
    # Add legend
    # ax.legend(frameon=False, loc='upper right', fontfamily="Palatino")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show
    plt.savefig('user4_priya_speeds.png')
    plt.savefig('user4_priya_speeds.pdf')
    plt.show()
    
    return all_speed_data

# Run the analysis
print("="*60)
print("USER 4 PRIYA SPEED ANALYSIS")
print("="*60)

speed_data = plot_user4_priya_speeds()




