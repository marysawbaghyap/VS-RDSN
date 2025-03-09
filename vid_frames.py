import cv2

def extract_frames(video_path, output_dir):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Initialize frame count
    frame_count = 0
    
    # Read the first frame
    success, frame = video_capture.read()
    
    # Loop through the video frames
    while success:
        # Construct frame name with leading zeros
        frame_name = f'{frame_count + 1:06}.jpg'
        
        # Write the frame to disk
        cv2.imwrite(f'{output_dir}/{frame_name}', frame)
        
        # Read the next frame
        success, frame = video_capture.read()
        
        # Increment frame count
        frame_count += 1
    
    # Release the video capture object
    video_capture.release()

# Example usage
video_path = 'Air_Force_One.mp4'  # Path to the input video file
output_dir = r"C:\Users\JEJEHOsuje\Downloads\vsum\frames"  # Directory to save the extracted frames
extract_frames(video_path, output_dir)
