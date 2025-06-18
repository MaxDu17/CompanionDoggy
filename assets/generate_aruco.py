import cv2
import numpy as np
import argparse

def generate_aruco_tag(dictionary_type, marker_id, marker_size, output_filename):
    # Load the specified ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    
    # Generate the marker
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    # Save the marker image
    cv2.imwrite(output_filename, marker_image)
    print(f"ArUco marker saved as {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an ArUco marker.")
    parser.add_argument("--dict", type=int, default=cv2.aruco.DICT_6X6_250, help="ArUco dictionary type (e.g., cv2.aruco.DICT_6X6_250)")
    parser.add_argument("--id", type=int, default=0, help="Marker ID")
    parser.add_argument("--size", type=int, default=500, help="Marker size in pixels")
    parser.add_argument("--output", type=str, default="aruco_marker.png", help="Output filename")
    
    args = parser.parse_args()
    
    generate_aruco_tag(args.dict, args.id, args.size, args.output)
