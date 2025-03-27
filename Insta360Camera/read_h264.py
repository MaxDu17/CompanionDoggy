import ffmpeg
import subprocess
import numpy as np 
import cv2 

def read_h264_stream_with_ffmpeg(stream_url):
    # Call ffmpeg using subprocess to read the H264 stream
    try:
        # Set up ffmpeg command to read the stream
        process = (
            ffmpeg
            .input(stream_url)
            .output('pipe:1', format='rawvideo', pix_fmt='bgr24')  # Output raw video to stdout
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        while True:
            # Read raw frames from stdout
            in_bytes = process.stdout.read(1440 * 2 * 1440 * 3)  # Assuming 1920x1080 resolution (modify as necessary)
            # in_bytes = process.stdout.read(1920 * 1080 * 3)  # Assuming 1920x1080 resolution (modify as necessary)
            array = np.frombuffer(in_bytes, dtype=np.uint8).reshape(1440, 1440*2, 3)
            cv2.imshow("test", array)
            cv2.waitKey(1) 
  

            if not in_bytes:
                print("No more data.")
                break

            # Process raw frame data (here we just print length, but you can decode and process the frame)
            print(f"Received frame of size: {len(in_bytes)} bytes")

            # Example: You can further process the raw frame bytes here
            # e.g., Decode with a library or save to a file

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        process.stdout.close()
        process.stderr.close()
        process.wait()

# Example usage
stream_url = '01.h264'
read_h264_stream_with_ffmpeg(stream_url)