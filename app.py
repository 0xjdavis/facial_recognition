import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import tempfile
import os
import sys
import traceback

# Function to check and install required libraries
def check_dependencies():
    try:
        import face_recognition
        return True
    except ImportError:
        st.error("""
        The 'face_recognition' library is not installed. 
        
        Please install it using:
        ```
        pip install face_recognition
        ```
        
        Note: This library requires dlib which may need additional system dependencies:
        - On Ubuntu/Debian: `sudo apt-get install -y build-essential cmake libopenblas-dev`
        - On Windows: Install Visual C++ Build Tools and CMake
        - On macOS: `brew install cmake`
        """)
        return False

# Check dependencies
import_success = check_dependencies()

# Only import face_recognition if dependency check passed
if import_success:
    import face_recognition

# Function to create a default mask
def create_default_mask():
    # Create a simple oval mask with transparency
    mask = np.zeros((300, 300, 4), dtype=np.uint8)
    center = (150, 150)
    axes = (120, 160)
    cv2.ellipse(mask, center, axes, 0, 0, 360, (0, 255, 0, 200), -1)  # Green oval with transparency
    return mask

# Function to apply mask to face
def apply_mask(frame, face_locations, mask_img):
    # Create a copy of the frame
    result_image = frame.copy()
    
    for face_location in face_locations:
        try:
            # Extract face location coordinates
            top, right, bottom, left = face_location
            face_width = right - left
            face_height = bottom - top
            
            # Skip faces that are too small
            if face_width <= 0 or face_height <= 0:
                continue
                
            # Get the face region
            face_roi = result_image[top:bottom, left:right]
            
            # Check if mask has alpha channel (4 channels)
            if mask_img.shape[2] == 4:
                # Resize the mask to match the face dimensions
                mask_resized = cv2.resize(mask_img, (face_width, face_height))
                
                # Split the mask into color and alpha channels
                mask_rgb = mask_resized[:, :, :3]
                mask_alpha = mask_resized[:, :, 3:4] / 255.0  # Keep dimension for broadcasting
                
                # Blend the face and mask
                blended = face_roi * (1 - mask_alpha) + mask_rgb * mask_alpha
                result_image[top:bottom, left:right] = blended.astype(np.uint8)
            else:
                # If no alpha channel, just use the mask with 50% opacity
                mask_resized = cv2.resize(mask_img, (face_width, face_height))
                result_image[top:bottom, left:right] = cv2.addWeighted(
                    face_roi, 0.5, mask_resized, 0.5, 0
                )
        except Exception as e:
            print(f"Error applying mask: {e}")
            continue
            
    return result_image

st.set_page_config(page_title="Facial Recognition and Mask App", layout="wide")

st.title("Facial Recognition and Mask Application")
st.write("This application captures video, performs facial recognition, and creates a facial mask from an uploaded image.")

# Create two columns for the interface
col1, col2 = st.columns(2)

# Initialize session state
if 'placeholder' not in st.session_state:
    st.session_state.placeholder = None
if 'stop' not in st.session_state:
    st.session_state.stop = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_time' not in st.session_state:
    st.session_state.last_time = 0
if 'fps' not in st.session_state:
    st.session_state.fps = 0

with col1:
    st.header("Video Capture and Face Recognition")
    
    # Button to start/stop video capture
    if st.button('Start/Stop Video'):
        st.session_state.stop = not st.session_state.stop
    
    # Create a placeholder for the video
    if st.session_state.placeholder is None:
        st.session_state.placeholder = st.empty()
    
    # Video capture options
    video_source = st.selectbox(
        "Select Video Source",
        ["Webcam", "Upload Video File"],
        index=0
    )
    
    # File uploader for video if user selects that option
    video_file = None
    if video_source == "Upload Video File":
        video_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])

    # Display FPS counter
    fps_text = st.empty()

with col2:
    st.header("Mask Settings")
    
    # Upload mask image
    mask_file = st.file_uploader("Upload mask image (with transparency)", type=['png', 'jpg', 'jpeg'])
    
    # If a mask is uploaded, display it
    if mask_file is not None:
        try:
            mask_image = Image.open(mask_file)
            st.image(mask_image, caption="Uploaded Mask", width=200)
            
            # Ensure image has alpha channel
            if mask_image.mode != 'RGBA':
                mask_image = mask_image.convert('RGBA')
                
            # Convert PIL image to numpy array for processing
            mask_np = np.array(mask_image)
            
            # Ensure mask has 4 channels (RGB + Alpha)
            if mask_np.shape[2] != 4:
                # Add alpha channel if missing
                alpha = np.ones(mask_np.shape[:2], dtype=np.uint8) * 255
                mask_np = np.dstack((mask_np, alpha))
        except Exception as e:
            st.error(f"Error processing mask image: {e}")
            # Fall back to default mask
            mask_np = create_default_mask()
    else:
        # Default mask (just a colored rectangle with transparency)
        mask_np = create_default_mask()
    
    # Facial recognition settings
    st.subheader("Face Recognition Settings")
    
    recognition_frequency = st.slider(
        "Recognition Frequency (frames)",
        min_value=1,
        max_value=30,
        value=5,
        help="How often to run facial recognition (every N frames)"
    )
    
    detection_confidence = st.slider(
        "Face Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Confidence threshold for face detection"
    )
    
    # Advanced settings (expandable)
    with st.expander("Advanced Settings"):
        mask_opacity = st.slider("Mask Opacity", 0.0, 1.0, 0.7, 0.1)
        show_face_boxes = st.checkbox("Show Face Bounding Boxes", value=True)
        flip_camera = st.checkbox("Flip Camera Horizontally", value=True)

# Main video processing function
def process_video():
    try:
        # Setup video capture
        if video_source == "Webcam":
            cap = cv2.VideoCapture(0)  # Use webcam
            # Try to set reasonable resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            if video_file is not None:
                # Save uploaded video to a temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(video_file.read())
                tfile_path = tfile.name
                tfile.close()
                cap = cv2.VideoCapture(tfile_path)
            else:
                st.warning("Please upload a video file.")
                return
        
        # Check if camera opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open video source.")
            return
        
        # Get face locations initially
        face_locations = []
        
        # Status indicator
        status_text = st.empty()
        
        # Main video loop
        while cap.isOpened() and not st.session_state.stop:
            try:
                # Read a frame
                ret, frame = cap.read()
                
                if not ret:
                    if video_source == "Upload Video File":
                        # For uploaded videos, loop back to beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        status_text.warning("Failed to capture video frame.")
                        time.sleep(1)  # Wait before retrying
                        continue
                
                # Flip frame if needed
                if flip_camera:
                    frame = cv2.flip(frame, 1)
                
                # Calculate FPS
                current_time = time.time()
                st.session_state.frame_count += 1
                
                if current_time - st.session_state.last_time >= 1.0:
                    st.session_state.fps = st.session_state.frame_count
                    st.session_state.frame_count = 0
                    st.session_state.last_time = current_time
                
                # Update FPS display
                fps_text.text(f"FPS: {st.session_state.fps}")
                
                # Run face recognition on every Nth frame
                if st.session_state.frame_count % recognition_frequency == 0:
                    # Convert BGR to RGB for face_recognition
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Find face locations
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                    
                    # Update status
                    if face_locations:
                        status_text.success(f"Detected {len(face_locations)} faces")
                    else:
                        status_text.info("No faces detected")
                
                # Display face boxes if option is enabled
                if show_face_boxes and face_locations:
                    for top, right, bottom, left in face_locations:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Apply mask to faces
                if face_locations and mask_np is not None:
                    try:
                        frame = apply_mask(frame, face_locations, mask_np)
                    except Exception as e:
                        status_text.error(f"Error applying mask: {e}")
                
                # Convert to RGB for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the resulting frame
                st.session_state.placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
            except Exception as e:
                status_text.error(f"Error processing frame: {e}")
                time.sleep(1)  # Add delay to avoid rapid error loops
        
        # Release video capture
        cap.release()
        
        # Clean up temporary file if used
        if video_source == "Upload Video File" and video_file is not None:
            try:
                os.unlink(tfile_path)
            except:
                pass
                
    except Exception as e:
        st.error(f"Error in video processing: {e}")

# Display app info
st.sidebar.header("About This App")
st.sidebar.write("""
This application captures video from your webcam or an uploaded video file, 
performs facial recognition, and applies a custom mask to detected faces.

To get started:
1. Upload a mask image (or use the default)
2. Select your video source
3. Click "Start/Stop Video" to begin
""")

# Instructions for troubleshooting
st.sidebar.header("Troubleshooting")
st.sidebar.write("""
If you encounter issues:
- Make sure your webcam is connected and accessible
- Try reducing the recognition frequency for better performance
- Ensure proper lighting for better face detection
- If using an uploaded mask, ensure it has transparency (PNG format)
""")

# Run the video processing function if dependency check passed and not stopped
if import_success and not st.session_state.stop:
    try:
        process_video()
    except Exception as e:
        st.error(f"Error: {e}")
        st.error(traceback.format_exc())
        st.info("Please check the troubleshooting tips in the sidebar.")
