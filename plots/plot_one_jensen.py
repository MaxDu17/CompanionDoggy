# Plot Jensen's session 3 data with target speed lines

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
                # Calculate area under the m/s curve (total distance)
                # Area = sum of speeds * time_interval (since each point represents average speed over time_interval)
        total_distance = sum(timed_speeds) * time_interval
        print(
            f"Area under m/s curve (total distance): {total_distance:.2f} meters ({total_distance / 1000:.2f} km)")
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(timed_speeds, 'b-', linewidth=2)
            plt.xlabel('Time Window Index')
            

            
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

def plot_jensen_session3_speeds():
    """
    Plot Jensen's session 3 speed data with target speed lines
    """
    # Jensen's session 3 file path
    # file_dir = "/Users/jennifergrannen/Documents/Stanford/iliad/CompanionDoggy/user_study_data/Jensen/"
    file_dir = "/Users/maxjdu/Documents/GitHub/CompanionDoggy/logs/jensen/"
    filepath = os.path.join(file_dir, "session3.gpx")
    
    print(f"Processing Jensen's session 3 data...")
    gpx_processor = GPXProcessor(filepath)
    
    # Get timed speeds (5-second intervals)
    timed_speeds = gpx_processor.get_timed_speeds(time_interval=5, plot=False, use_mph=False)
    
    # Take first 80 seconds of data (16 data points at 5-second intervals)
    # end_index = min(16, len(timed_speeds))
    end_index = int(min(3.4*60*4/5, len(timed_speeds)))
    jensen_speeds = timed_speeds[:end_index]
    
    print(f"Jensen session 3: {len(jensen_speeds)} data points (80 seconds)")
    
    # Create the plot
    # fig, ax = plt.subplots(figsize=(5, 4))
    # fig, ax = plt.subplots(figsize=(4, 3))
    # fig, ax = plt.subplots(figsize=(6, 3))
    fig, ax = plt.subplots(figsize=(7, 3))

    
    # Time points (5-second intervals) in minutes
    time_points = np.arange(0, len(jensen_speeds) * 5, 5) / 60  # Convert to minutes
    
    # Plot Jensen's speeds
    ax.plot(time_points, jensen_speeds, '-', color='mediumseagreen', 
           linewidth=2, label="Jensen's Speed", markersize=4)
    
    # Set x-axis ticks to go to the end of the data with whole numbers
    max_time_minutes = len(jensen_speeds) * 5 / 60
    max_whole_minutes = int(np.ceil(max_time_minutes))
    x_ticks = np.arange(0, max_whole_minutes + 1, 1)  # Whole number ticks
    ax.set_xticks(x_ticks)
    
    # Create labels for every other tick
    x_labels = [f'{int(t)}' if t % 2 == 0 else '' for t in x_ticks]
    ax.set_xticklabels(x_labels, font="Palatino")
    
    # Set y-axis ticks
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels([0, 1, 2, 3, 4, 5], font="Palatino")
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add alternating target speed line (fast-slow-fast pattern every 30 seconds)
    fast_target = 2.5  # m/s
    slow_target = 1.8  # m/s
    
    # Create time and target arrays for the alternating pattern (in minutes)
    target_times = []
    target_speeds = []
    
    # Convert 3.4 minutes to seconds for calculations, then back to minutes
    interval_minutes = 3.4
    
    # 0-3.4min: Fast (red)
    target_times.extend([0, interval_minutes])
    target_speeds.extend([fast_target, fast_target])
    
    # 3.4-6.8min: Slow (green) 
    target_times.extend([interval_minutes, interval_minutes*2])
    target_speeds.extend([slow_target, slow_target])
    
    # 6.8-10.2min: Fast (red)
    target_times.extend([interval_minutes*2, interval_minutes*3])
    target_speeds.extend([fast_target, fast_target])

    # 10.2-13.6min: slow (green)
    target_times.extend([interval_minutes*3, interval_minutes*4])
    target_speeds.extend([slow_target, slow_target])
    
    # # 30-60s: Fast (red) 
    # target_times.extend([3.4*60*4, 3.4*60*5])
    # target_speeds.extend([fast_target, fast_target])
    
    # # 60-80s: slow (green)
    # target_times.extend([3.4*60*5, 3.4*60*6])
    # target_speeds.extend([slow_target, slow_target])
    
    # Plot the alternating target line
    ax.plot(target_times, target_speeds, '--', color='red', alpha=0.7, linewidth=2, 
           label='Target Pattern')

    # Change y-axis tick label font size
    ax.tick_params(axis="y", labelsize=12)

    # Change x-axis tick label font size
    ax.tick_params(axis="x", labelsize=12)

    # Add labels and title
    # ax.set_xlabel('Time (minutes)', fontsize=12, font="Palatino")
    # ax.set_ylabel('Speed (m/s)', fontsize=12, font="Palatino")
    # ax.set_title("Jensen's Session 3 - Speed Over Time", fontsize=14, font="Palatino", pad=20)
    
    # Add legend
    # ax.legend(frameon=False, loc='upper right', fontfamily="Palatino")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show
    plt.savefig('jensen_session3_speeds.png')
    plt.savefig('jensen_session3_speeds.pdf')
    plt.show()
    
    return jensen_speeds

# Run the analysis
print("="*60)
print("JENSEN SESSION 3 SPEED ANALYSIS")
print("="*60)

speed_data = plot_jensen_session3_speeds()
