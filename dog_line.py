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

def plot_contours_interactive(frame, contours, roi_offset, view_bounds_list, min_contour_length, min_contour_area):
    """Interactive function to display contours and show information about selected ones."""
    # Create a copy of the frame for visualization
    vis_frame = frame.copy()
    
    # Store contour information for later use
    contour_info = []
    
    # Draw all contours without filtering
    for i, contour in enumerate(contours):
        # Adjust contour points for ROI offset
        adjusted_contour = contour.copy()
        adjusted_contour[:, :, 1] += roi_offset
        
        # Draw the contour in yellow
        cv2.drawContours(vis_frame, [adjusted_contour], 0, (255, 255, 0), 5)
        
        # Get points that are within the view window
        points_in_view = []
        for point in adjusted_contour:
            x, y = point[0]
            if (view_bounds_list[0][0] <= x <= view_bounds_list[0][1] and 
                view_bounds_list[0][2] <= y <= view_bounds_list[0][3]):
                points_in_view.append([x, y])
        
        # Store contour information if we have enough points
        if len(points_in_view) >= 5:
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
            
            # Calculate angle
            angle = float(np.arctan2(vy, vx) * 180 / np.pi)
            
            contour_info.append({
                'contour': adjusted_contour,
                'index': i,
                'length': len(contour),
                'area': cv2.contourArea(contour),
                'angle': angle,
                'center': (int(x), int(y)),
                'vx': float(vx),
                'vy': float(vy),
                'points_in_view': points_in_view
            })
    
    # Create window and set mouse callback
    cv2.namedWindow('Contour Selection')
    selected_contour = [None]  # Using list to store selected contour (to modify in callback)
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is inside any contour
            for info in contour_info:
                if cv2.pointPolygonTest(info['contour'], (x, y), False) >= 0:
                    selected_contour[0] = info
                    break
    
    cv2.setMouseCallback('Contour Selection', mouse_callback)
    
    while True:
        # Create a copy of the visualization frame
        display_frame = vis_frame.copy()
        
        # If a contour is selected, highlight it and show information
        if selected_contour[0] is not None:
            info = selected_contour[0]
            # Draw selected contour in red
            cv2.drawContours(display_frame, [info['contour']], 0, (0, 0, 255), 5)
            
            # Draw points used for PCA in green
            for point in info['points_in_view']:
                cv2.circle(display_frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
            
            # Draw PCA direction vector in blue
            center = info['center']
            vx, vy = info['vx'], info['vy']
            cv2.arrowedLine(display_frame, 
                          center,
                          (int(center[0] + vx*50), int(center[1] + vy*50)),
                          (255, 0, 0), 2)
            
            # Display information
            info_text = [
                f"Contour {info['index']}",
                f"Length: {info['length']} points",
                f"Area: {info['area']:.2f}",
                f"Angle: {info['angle']:.2f} degrees",
                f"Center: {center}",
                f"Points in view: {len(info['points_in_view'])}"
            ]
            
            # Draw text background
            for i, text in enumerate(info_text):
                cv2.putText(display_frame, text, (10, 30 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(display_frame, text, (10, 30 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Contour Selection', display_frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

def detect_line(video_path):
    # Read the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    output_path = video_path.rsplit('.', 1)[0] + '_processed.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Fisheye correction parameters
    K = np.array([[width, 0, width/2],
                  [0, height, height/2],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Distortion coefficients
    
    # Define view area
    view_x = width // 2  # center x position of view area
    view_y = height // 2 + VIEW_Y_OFFSET  # center y position of view area

    # Define target point (where we want the line to pass through)
    target_x = width // 2  # center x position
    target_y = height // 2 + TARGET_Y_OFFSET  # center y position
    
    # Calculate view bounds
    view_bounds_list = []
    
    # View area bounds
    view_left = view_x - VIEW_WIDTH // 2
    view_right = view_x + VIEW_WIDTH // 2
    view_top = view_y - VIEW_HEIGHT // 2
    view_bottom = view_y + VIEW_HEIGHT // 2
    view_bounds_list.append([view_left, view_right, view_top, view_bottom])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply fisheye correction
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (width, height), cv2.CV_32FC1)
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # only look for the line in the bottom half of the image
        roi = gray[int(gray.shape[0] * 1/2):, :]
        
        # Apply adaptive thresholding to better detect the line
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations to clean up the line
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
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
                            (view_left, view_top),
                            (view_right, view_bottom),
                            255, -1)
                
                # Calculate overlap area
                overlap = cv2.bitwise_and(contour_mask, view_mask)
                overlap_area = cv2.countNonZero(overlap)
                
                # Skip contours with insufficient overlap
                if overlap_area < MIN_OVERLAP_AREA:
                    continue
                
                # Get points that are within the view window
                points_in_view = []
                for point in adjusted_contour:
                    x, y = point[0]
                    if (view_left <= x <= view_right and 
                        view_top <= y <= view_bottom):
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
                    score_info = score_line(vx, vy, x, y, target_x, target_y)
                    
                    # Plot this line in yellow
                    plot_line_and_target(vis_frame, vx, vy, x, y, target_x, target_y, 
                                       view_bounds_list, color=(0, 255, 255))
                    
                    if score_info['score'] < best_score:
                        best_score = score_info['score']
                        best_line = [vx, vy, x, y]
                        best_score_info = score_info
            
            if best_line is not None:
                # Plot the best line in red
                plot_line_and_target(vis_frame, best_line[0], best_line[1], best_line[2], best_line[3], 
                                   target_x, target_y, view_bounds_list, color=(0, 0, 255))
                
                # Print line information
                print(f"Distance from target: {best_score_info['point_to_line_distance']:.2f} pixels, "
                      f"Angle: {best_score_info['angle']:.2f} degrees, "
                      f"Y at target x: {best_score_info['line_y_at_target']:.2f}")
            
            # Write the frame to the output video
            out.write(vis_frame)
            
            # Display the visualization
            cv2.imshow('Line Detection', vis_frame)
            
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Release the video capture and close all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processed video saved to: {output_path}")

# Example usage
video_path = '/Users/jennifergrannen/Downloads/only_line.mp4'
detect_line(video_path)
