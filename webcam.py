import cv2

# Open the webcam (in this case, the default webcam)
video_capture = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not video_capture.isOpened():
    print("Error: Unable to open webcam.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640, 480))

# Capture video from the webcam and save it
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Write the frame into the output video
    output_video.write(frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects, and close OpenCV windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
