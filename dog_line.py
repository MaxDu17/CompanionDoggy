# write a python script that will take a mp4 file and use opencv to detect the line in the bottom of the video
# you should use greyscale and hough lines to detect the line
# you should plot the line on the video
# you should save the video with the line plotted

import cv2
import time
import numpy as np
from colors import BLUE_LINE, WHITE_LINE

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

# Color adjustment parameters
def nothing(x):
    pass

def preload_color_ranges(color_type):
    """Return BGR color ranges for blue or white line detection."""
    if color_type.lower() == "blue":
        # Blue line ranges (BGR format)
        lower_bgr = BLUE_LINE[0]
        upper_bgr = BLUE_LINE[1]
    elif color_type.lower() == "white":
        # White line ranges (BGR format)
        lower_bgr = WHITE_LINE[0]
        upper_bgr = WHITE_LINE[1]
    else:
        # Default to blue if unknown
        lower_bgr = BLUE_LINE[0]
        upper_bgr = BLUE_LINE[1]
    
    return lower_bgr, upper_bgr

def create_color_sliders(preload_color="blue"):
    """Create a window with sliders for adjusting RGB color ranges."""
    cv2.namedWindow('Color Adjustments', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Color Adjustments', 600, 50) 
    
    # Position the color slider window at the top
    cv2.moveWindow('Color Adjustments', 0, 0)
    
    # Create trackbars for RGB lower bounds
    cv2.createTrackbar('R_min', 'Color Adjustments', 0, 255, nothing)
    cv2.createTrackbar('G_min', 'Color Adjustments', 0, 255, nothing)
    cv2.createTrackbar('B_min', 'Color Adjustments', 0, 255, nothing)
    
    # Create trackbars for RGB upper bounds
    cv2.createTrackbar('R_max', 'Color Adjustments', 255, 255, nothing)
    cv2.createTrackbar('G_max', 'Color Adjustments', 255, 255, nothing)
    cv2.createTrackbar('B_max', 'Color Adjustments', 255, 255, nothing)
    
    # Get preloaded color ranges
    lower_bgr, upper_bgr = preload_color_ranges(preload_color)
    
    # Set initial values based on preloaded colors (BGR format in OpenCV)
    cv2.setTrackbarPos('B_min', 'Color Adjustments', int(lower_bgr[0]))  # Blue minimum
    cv2.setTrackbarPos('B_max', 'Color Adjustments', int(upper_bgr[0]))  # Blue maximum
    cv2.setTrackbarPos('G_min', 'Color Adjustments', int(lower_bgr[1]))  # Green minimum
    cv2.setTrackbarPos('G_max', 'Color Adjustments', int(upper_bgr[1]))  # Green maximum
    cv2.setTrackbarPos('R_min', 'Color Adjustments', int(lower_bgr[2]))  # Red minimum
    cv2.setTrackbarPos('R_max', 'Color Adjustments', int(upper_bgr[2]))  # Red maximum

def get_current_color_range():
    """Get the current RGB color range from the sliders."""
    r_min = cv2.getTrackbarPos('R_min', 'Color Adjustments')
    g_min = cv2.getTrackbarPos('G_min', 'Color Adjustments')
    b_min = cv2.getTrackbarPos('B_min', 'Color Adjustments')
    r_max = cv2.getTrackbarPos('R_max', 'Color Adjustments')
    g_max = cv2.getTrackbarPos('G_max', 'Color Adjustments')
    b_max = cv2.getTrackbarPos('B_max', 'Color Adjustments')
    
    lower_bgr = np.array([b_min, g_min, r_min])  # BGR format in OpenCV
    upper_bgr = np.array([b_max, g_max, r_max])  # BGR format in OpenCV
    
    return lower_bgr, upper_bgr

# consistency is important
def score_line(vx, vy, x, y, target_x, target_y):
    """Score a line based on its distance to target and angle from vertical."""
    # Calculate angle and angle penalty
    angle = float(np.arctan2(vy, vx) * 180 / np.pi)
   
    angle_diff = abs(angle - (90))  # Difference from vertical
    # print("----")
    # print(vy, vx, angle)
    # print(angle_diff)
    # print("------")
    angle_penalty = max(0, angle_diff - ANGLE_THRESHOLD) * ANGLE_PENALTY_MULTIPLIER

    line_y_at_target = target_y 
    line_x_at_target = target_x # default value  
    if vx > 1e-6: # has a horizontal component 
        line_y_at_target = y + (target_x - x) * (vy / vx) # where the line intersects with the x value of the target 

    if vy > 1e-6: # this line has a vertical component 
        line_x_at_target = (target_y - y) * (vx / vy) + x # where the line intersects with the y value of the target

    # Calculate actual distance from target point to line
    # Using point-to-line distance formula
    a = vy
    b = -vx
    c = vx * y - vy * x
    point_to_line_distance = abs(a * target_x + b * target_y + c) / np.sqrt(a * a + b * b)

    # Score combines point-to-line distance and angle penalty
    score = point_to_line_distance + angle_penalty

    return {
        'score': score,
        'angle': angle,
        'point_to_line_distance': point_to_line_distance,
        'angle_penalty': angle_penalty,
        'line_y_at_target': line_y_at_target,
        "line_x_at_target": line_x_at_target
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
        lefty = int((-x * vy / vx) + y)
        righty = int(((frame.shape[1] - x) * vy / vx) + y)
        leftx = 0
        rightx = frame.shape[1] - 1

    # Draw the line
    cv2.line(frame, (rightx, righty),
             (leftx, lefty), color, 2)

    # Draw target point
    cv2.circle(frame, (target_x, target_y), 5, (0, 0, 255), -1)

    # Draw the point (x,y) in green
    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Draw the direction vector in blue
    # cv2.arrowedLine(frame,
    #                 (int(x), int(y)),
    #                 (int(x + vx * 50), int(y + vy * 50)),
    #                 (255, 0, 0), 2)

    # Draw view rectangles
    # for i, view_bounds in enumerate(view_bounds_list):
    #     # Use different colors for different view areas
    #     rect_color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for first, Red for second
    #     cv2.rectangle(frame,
    #                   (view_bounds[0], view_bounds[2]),  # top-left
    #                   (view_bounds[1], view_bounds[3]),  # bottom-right
    #                   rect_color, 2)

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
        # self.view_bounds_list = []
        
        # View area bounds
        self.view_left = view_x - VIEW_WIDTH // 2
        self.view_right = view_x + VIEW_WIDTH // 2
        self.view_top = view_y - VIEW_HEIGHT // 2
        self.view_bottom = view_y + VIEW_HEIGHT // 2
        # self.view_bounds_list.append([self.view_left, self.view_right, self.view_top, self.view_bottom])
        
        self.view_bounds = [self.view_left, self.view_right, self.view_top, self.view_bottom]
        self.K = K
        self.D = D

        self.preload_colors = "blue" # or "white"

        # self.past_lines = # set a list of past lines parameteied by angle and something else

    def get_principle_axis(self, points_in_view):
        # Center the points
        mean = np.mean(points_in_view, axis=0)
        centered_points = points_in_view - mean

        # Compute covariance matrix
        cov = np.cov(centered_points.T)

        # Get eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Get the principal direction (eigenvector with largest eigenvalue)
        principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
        if principal_direction[1] < 0: # always have the vector pointing upwards 
            return -principal_direction 
        return principal_direction


    def detect_line(self, frame):
        # Apply fisheye correction
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, (self.width, self.height), cv2.CV_32FC1)
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop to the green box area
        cropped_frame = frame[self.view_top:self.view_bottom, self.view_left:self.view_right]

        # test_colors(cropped_frame)

        # Get current color range from sliders
        lower_bgr, upper_bgr = get_current_color_range()
        
        # Create mask using dynamic BGR range
        mask = cv2.inRange(cropped_frame, lower_bgr, upper_bgr)

        # Apply mask to get only the light colored regions
        masked_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)
        # cv2.imshow('masked_frame', masked_frame)
        cv2.moveWindow("masked_frame", 820, 120)  

        # Create a blue background for the masked frame
        blue_bg = np.zeros_like(masked_frame)
        blue_bg[:] = (0, 0, 0)  # Blue in BGR

        # Create a mask for the non-black pixels in masked_frame
        gray_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        _, non_black_mask = cv2.threshold(gray_masked, 1, 255, cv2.THRESH_BINARY)
        non_black_mask = cv2.cvtColor(non_black_mask, cv2.COLOR_GRAY2BGR) / 255.0
        # Combine masked frame with blue background
        masked_with_blue = (1 - non_black_mask) * blue_bg + non_black_mask * masked_frame
        # Convert to uint8 for video writing
        masked_with_blue = masked_with_blue.astype(np.uint8)
        # Create overlay frame
        overlay = np.zeros_like(frame)

        # Place the masked frame with blue background in the upper right corner
        overlay[0:masked_frame.shape[0], self.width - masked_frame.shape[1]:self.width] = masked_with_blue
        vis_frame = frame.copy()
        # Replace the upper right corner with the masked frame
        vis_frame[0:masked_frame.shape[0], self.width - masked_frame.shape[1]:self.width] = masked_with_blue
        # ^^ all this is for the visualization


        # Convert to grayscale for contour detection
        cropped_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to better detect the line
        thresh = cv2.adaptiveThreshold(cropped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Apply morphological operations to clean up the line
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cv2.rectangle(vis_frame,
                (self.view_bounds[0], self.view_bounds[2]),  # top-left
                (self.view_bounds[1], self.view_bounds[3]),  # bottom-right
                (0, 0, 255), 2)

            return {"success": False, "message": "No contours found", "frame": vis_frame.copy()}

        best_score = float('inf')
        best_line = None

        # Process each contour
        for contour in contours:
            # Get points that are within the view window
            # if cv2.contourArea(contour) < 400:
            #     continue # don't consider small contours
            points_in_view = []

            for point in contour:
                x, y = point[0]
                points_in_view.append([x, y])

            # Only proceed if we have enough points
            if len(points_in_view) >= MIN_POINTS_FOR_PCA:
                points_in_view = np.array(points_in_view)
                mean = np.mean(points_in_view, axis=0)

                vx, vy = self.get_principle_axis(points_in_view)
                x, y = mean

                # Adjust coordinates back to original image space
                x += self.view_left
                y += self.view_top

                # Calculate distance from tracked point to this contour's center
                dist_to_tracked = np.sqrt((x - self.target_x) ** 2 + (y - self.target_y) ** 2)

                # Score the line
                score_info = score_line(vx, vy, x, y, self.target_x, self.target_y)
                # plot_line_and_target(vis_frame, vx, vy, x, y,
                #                      self.target_x, self.target_y, self.view_bounds_list, color=(255, 0, 0)) #, alpha = 0.1)

                # Add distance penalty to score
                score_info['score'] += dist_to_tracked

                if score_info['score'] < best_score:
                    best_score = score_info['score']
                    best_line = [vx, vy, x, y]
                    best_score_info = score_info
                    best_contour = contour  # Store the best contour
        if best_line is None:
            return {"success": False, "message": "No best line found", "frame": vis_frame}
        # Draw only the best contour in yellow (adjusted to original image space)
        adjusted_contour = best_contour.copy()
        adjusted_contour[:, :, 0] += self.view_left
        adjusted_contour[:, :, 1] += self.view_top

        cv2.drawContours(vis_frame, [adjusted_contour], 0, (255, 255, 0), 5)
        # cv2.putText(vis_frame, str(cv2.contourArea(adjusted_contour)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4,
        #             cv2.LINE_AA)
        # Plot the best line in red
        plot_line_and_target(vis_frame, best_line[0], best_line[1], best_line[2], best_line[3],
                             self.target_x, self.target_y, self.view_bounds, color=(0, 0, 255))
        cv2.rectangle(vis_frame,
                    (self.view_bounds[0], self.view_bounds[2]),  # top-left
                    (self.view_bounds[1], self.view_bounds[3]),  # bottom-right
                    (255, 0, 0), 2)


        if np.isinf(best_score_info["line_x_at_target"]): # if line is perpendicular then return no horizontal translation; focus on the angle 
            return {"success": True, "frame": vis_frame, "best_line": best_line, "message": "Success", "angle" : best_score_info['angle'], "x_at_target" : 0}

        cv2.circle(vis_frame, (int(best_score_info["line_x_at_target"]), self.target_y), 5, (255, 0, 255), -1)
        x_at_target = best_score_info["line_x_at_target"] - self.target_x
        return {"success": True, "frame": vis_frame, "best_line": best_line, "message": "Success", "angle" : best_score_info['angle'], "x_at_target" : x_at_target}

if __name__ == "__main__":
    video_example_path = "/Users/jennifergrannen/Downloads/only_line.mp4"

    cap = cv2.VideoCapture(video_example_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    K = np.array([[width, 0, width / 2],
                  [0, height, height / 2],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Distortion coefficients

    line_detector = LineDetector(
        width=width, height=height,
        K=K,
        D=D
    )
    
    # Create color adjustment sliders with preloaded colors
    create_color_sliders(line_detector.preload_colors)
    
    # Position the video window below the color slider
    cv2.namedWindow("Visual", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Visual", 100, 120)  # Position below the color slider

    output_path = video_example_path.rsplit('.', 1)[0] + '_processed_white.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Color Adjustment Controls:")
    print("- Use the 'Color Adjustments' window to adjust RGB ranges")
    print("- R: Red (0-255)")
    print("- G: Green (0-255)")
    print("- B: Blue (0-255)")
    print("- Press 'ESC' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        info = line_detector.detect_line(frame)
        if not info["success"]:
            # show the error message on the camera
            cv2.putText(info["frame"], info["message"], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4,
                        cv2.LINE_AA)
            cv2.imshow("Visual", info["frame"])
            cv2.waitKey(1)  # this is to allow the frame to be shown
            continue

        best_line = info["best_line"]
        cv2.imshow("Visual", info["frame"])
        out.write(info["frame"])
        if cv2.waitKey(1) == 27:
            break

        print(info["angle"], info["x_at_target"])
    out.release()
    cv2.destroyAllWindows()

