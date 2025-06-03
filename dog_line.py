# write a python script that will take a mp4 file and use opencv to detect the line in the bottom of the video
# you should use greyscale and hough lines to detect the line
# you should plot the line on the video
# you should save the video with the line plotted

import cv2
import time
import numpy as np

# View window parameters
VIEW_WIDTH = 250  # width of view area
VIEW_HEIGHT = 80  # height of view area
VIEW_Y_OFFSET = 80  # offset from center for view area

# Target point parameters
TARGET_Y_OFFSET = 80  # offset from center for target point

# Contour filtering parameters
MIN_OVERLAP_AREA = 500  # minimum area overlap with view window
MIN_POINTS_FOR_PCA = 5  # minimum points needed for PCA line fitting

# Line scoring parameters
ANGLE_THRESHOLD = 20  # degrees from vertical before penalty starts
ANGLE_PENALTY_MULTIPLIER = 5  # multiplier for angle penalty

def score_line(vx, vy, x, y, target_x, target_y):
    """Score a line based on its distance to target and angle from vertical."""
    # Calculate angle and angle penalty
    angle = float(np.arctan2(vy, vx) * 180 / np.pi)
    angle_diff = abs(angle - (90))  # Difference from vertical
    angle_penalty = max(0, angle_diff - ANGLE_THRESHOLD) * ANGLE_PENALTY_MULTIPLIER
    
    # Calculate where the line intersects the target x-coordinate
    line_y_at_target = y + (target_x - x) * (vy/vx)
    
    # Calculate actual distance from target point to line
    # Using point-to-line distance formula
    a = vy
    b = -vx
    c = vx*y - vy*x
    point_to_line_distance = abs(a*target_x + b*target_y + c) / np.sqrt(a*a + b*b)
    
    # Score combines point-to-line distance and angle penalty
    score = point_to_line_distance + angle_penalty
    
    return {
        'score': score,
        'angle': angle,
        'point_to_line_distance': point_to_line_distance,
        'angle_penalty': angle_penalty,
        'line_y_at_target': line_y_at_target
    }

def plot_line_and_target(frame, vx, vy, x, y, target_x, target_y, view_bounds_list, color=(0, 0, 255)):
    """Plot a line and target point on the frame."""
    # Calculate line endpoints
    if abs(vx) < 1e-6:  # If line is nearly vertical
        # For vertical lines, use the x coordinate and frame height
        lefty = 0
        righty = frame.shape[0]
        leftx = rightx = int(x)
    else:
        # For non-vertical lines, calculate y at left and right edges
        lefty = int((-x*vy/vx) + y)
        righty = int(((frame.shape[1]-x)*vy/vx)+y)
        leftx = 0
        rightx = frame.shape[1]-1
    
    # Draw the line
    cv2.line(frame, (rightx, righty), 
            (leftx, lefty), color, 2)
    
    # Draw target point
    cv2.circle(frame, (target_x, target_y), 5, (0, 0, 255), -1)
    
    # Draw the point (x,y) in green
    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # Draw the direction vector in blue
    cv2.arrowedLine(frame, 
                   (int(x), int(y)),
                   (int(x + vx*50), int(y + vy*50)),
                   (255, 0, 0), 2)
    
    # Draw view rectangles
    for i, view_bounds in enumerate(view_bounds_list):
        # Use different colors for different view areas
        rect_color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for first, Red for second
        cv2.rectangle(frame, 
                     (view_bounds[0], view_bounds[2]),  # top-left
                     (view_bounds[1], view_bounds[3]),  # bottom-right
                     rect_color, 2)
    
    return frame


class LineDetector:
    def __init__(self, width, height, K, D):
        # Define view area
        view_x = width // 2  # center x position of view area
        view_y = height // 2 + VIEW_Y_OFFSET  # center y position of view area

        self.width = width 
        self.height = height 
        # Define target point (where we want the line to pass through)
        self.target_x = width // 2  # center x position
        self.target_y = height // 2 + TARGET_Y_OFFSET  # center y position
        
        # Calculate view bounds
        self.view_bounds_list = []
        
        # View area bounds
        self.view_left = view_x - VIEW_WIDTH // 2
        self.view_right = view_x + VIEW_WIDTH // 2
        self.view_top = view_y - VIEW_HEIGHT // 2
        self.view_bottom = view_y + VIEW_HEIGHT // 2
        self.view_bounds_list.append([self.view_left, self.view_right, self.view_top, self.view_bottom])

        self.K = K
        self.D = D 


    def detect_line(self, frame):
        # Apply fisheye correction
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, (self.width, self.height), cv2.CV_32FC1)
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # only look for the line in the bottom half of the image
        roi = gray[int(gray.shape[0] * 1/2):, :]
        
        # Apply adaptive thresholding to better detect the line
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        # TODO: make these adjustable in sliders 
        
        # Apply morphological operations to clean up the line
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: 
            return {"success" : False, "message" : "No contours found", "frame" : frame.copy()}
    
        best_score = float('inf')
        best_line = None
        
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Process each contour
        for contour in contours:
            # Adjust contour points for ROI offset
            adjusted_contour = contour.copy()
            adjusted_contour[:, :, 1] += int(frame.shape[0] * 1/2)
            
            # Create a mask for the contour
            contour_mask = np.zeros_like(gray)
            cv2.drawContours(contour_mask, [adjusted_contour], 0, 255, -1)
            
            # Create a mask for the view window
            view_mask = np.zeros_like(gray)
            cv2.rectangle(view_mask, 
                        (self.view_left, self.view_top),
                        (self.view_right, self.view_bottom),
                        255, -1)
            
            # Calculate overlap area -> after this, it should have contours within the relevant area 
            overlap = cv2.bitwise_and(contour_mask, view_mask)
            overlap_area = cv2.countNonZero(overlap)
            
            # Skip contours with insufficient overlap
            if overlap_area < MIN_OVERLAP_AREA:
                continue
            
            # Get points that are within the view window
            points_in_view = []
            for point in adjusted_contour:
                x, y = point[0]
                if (self.view_left <= x <= self.view_right and 
                    self.view_top <= y <= self.view_bottom):
                    points_in_view.append([x, y])
            
            # Only proceed if we have enough points
            if len(points_in_view) >= MIN_POINTS_FOR_PCA:
                points_in_view = np.array(points_in_view)
                
                # Center the points
                mean = np.mean(points_in_view, axis=0)
                centered_points = points_in_view - mean
                
                # Compute covariance matrix
                cov = np.cov(centered_points.T)
                
                # Get eigenvectors and eigenvalues
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # Get the principal direction (eigenvector with largest eigenvalue)
                principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
                
                # Calculate the line parameters
                vx, vy = principal_direction
                x, y = mean
                
                # Draw the contour in yellow
                cv2.drawContours(vis_frame, [adjusted_contour], 0, (255, 255, 0), 5)
                
                # Score the line
                score_info = score_line(vx, vy, x, y, self.target_x, self.target_y)
                
                # Plot this line in yellow
                plot_line_and_target(vis_frame, vx, vy, x, y, self.target_x, self.target_y, 
                                    self.view_bounds_list, color=(0, 255, 255))
                
                if score_info['score'] < best_score:
                    best_score = score_info['score']
                    best_line = [vx, vy, x, y]
                    best_score_info = score_info
        
        if best_line is None: 
            return {"success" : False, "message" : "No best line found", "frame" : vis_frame}

        # Plot the best line in red
        plot_line_and_target(vis_frame, best_line[0], best_line[1], best_line[2], best_line[3], 
                            self.target_x, self.target_y, self.view_bounds_list, color=(0, 0, 255))
        
        # Print line information
        print(f"Distance from target: {best_score_info['point_to_line_distance']:.2f} pixels, "
                f"Angle: {best_score_info['angle']:.2f} degrees, "
                f"Y at target x: {best_score_info['line_y_at_target']:.2f}")
        return {"success": True, "frame": vis_frame, "best_line": best_line}
    


# # Example usage
# video_path = '/Users/jennifergrannen/Downloads/only_line.mp4'



# detect_line(video_path)
