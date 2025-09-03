#!/bin/bash
cd /home/max/CompanionDoggy/Insta360Camera/
# sudo ./bin/main


# Function to handle Ctrl+C (SIGINT)
cleanup() {
    echo "Caught Ctrl+C. Exiting..."
    exit 0
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT
FILE="/tmp/camera_ready"
# Infinite loop
while true; do
    # Run the script
    sudo ./bin/main

    # Check if the file exists
    if [ -f "$FILE" ]; then
        echo "Deleting $FILE..."
        rm "$FILE"
        echo "File deleted."
    else
        echo "File does not exist: $FILE"
    fi
    # Wait for 5 seconds
    sleep 0.1
done
