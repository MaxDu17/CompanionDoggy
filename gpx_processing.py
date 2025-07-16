# write a script to process gpx files and be able to get the instantaneous speed and plot that

import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
from datetime import datetime
import math

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

    def get_instantaneous_speed(self):
        # get the instantaneous speed
        instantaneous_speed = []
        time_diffs = []
        for track in self.gpx_data.tracks:
            for segment in track.segments:
                points = segment.points
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
        return instantaneous_speed, time_diffs

    def get_timed_speeds(self, time_interval=60*5, plot=False):
        instantaneous_speed, time_diffs = self.get_instantaneous_speed()
        timed_speeds = []
        agg_time_diff = 0
        for i in range(len(instantaneous_speed)):
            agg_time_diff += time_diffs[i]
            if agg_time_diff > time_interval:
                timed_speeds.append(instantaneous_speed[i])
                agg_time_diff = 0
        if plot:
            plt.plot(timed_speeds)
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
            plt.scatter(x_positions, y_positions)
            plt.show()
        return x_positions, y_positions

filepath = "/Users/jennifergrannen/Downloads/Evening_Walk.gpx"
gpx_processor = GPXProcessor(filepath)

timed_speeds = gpx_processor.get_timed_speeds(plot=True)




