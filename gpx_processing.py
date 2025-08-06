# write a script to process gpx files and be able to get the instantaneous speed and plot that

import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
from datetime import datetime
import math
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
        for track in gpx.tracks:
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

# filepath = "/Users/jennifergrannen/Downloads/Evening_Walk.gpx"
# filepath = "/Users/jennifergrannen/Downloads/Robot_Dog_calibration_walk.gpx"
filepath = "/Users/jennifergrannen/Downloads/Afternoon_Run.gpx"
gpx_processor = GPXProcessor(filepath)

timed_speeds = gpx_processor.get_timed_speeds(plot=True, use_mph=True)




