# Hand Gesture Recognition

This project implements a real-time hand gesture recognition system using a webcam. The system segments the hand region from the background and displays the thresholded hand image and the contours of the hand. Additionally, it allows saving segmented hand images by pressing the "s" key.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Description](#description)
- [References](#references)

## Requirements
- Python 3.x
- OpenCV
- imutils
- numpy
- scikit-learn

## Usage
1. Run the script:
    ```bash
    python hand_gesture_recognition.py
    ```
2. The system will start capturing video from the webcam.
3. Wait for the calibration to complete. The system will display `[STATUS] please wait! calibrating...` followed by `[STATUS] calibration successful...`.
4. The segmented hand region will be displayed in a separate window.
5. Press the "s" key to save the thresholded hand image.
6. Press the "q" key to exit the application.

## Description
The script captures video frames from the webcam and processes them to segment the hand region from the background. Here's a brief explanation of the main components of the script:

- **Background Averaging**: The function `run_avg()` computes the running average of the background to be used for background subtraction.
- **Hand Segmentation**: The function `segment()` performs background subtraction, applies thresholding, and finds contours to segment the hand region.
- **Main Loop**: The main loop captures video frames, processes them to segment the hand, and displays the thresholded hand image and contours. It also handles key presses for saving images and exiting the application.

### Script Workflow
1. **Initialize Variables**: Set initial parameters and configurations.
2. **Capture Video**: Start capturing video from the webcam.
3. **Resize and Flip Frame**: Resize the frame for consistent processing and flip it to create a mirror effect.
4. **Region of Interest (ROI)**: Define and extract the region of interest where the hand is expected to be.
5. **Convert to Grayscale and Blur**: Convert the ROI to grayscale and apply Gaussian blur to reduce noise.
6. **Background Calibration**: Perform background averaging for the first 30 frames to calibrate the background model.
7. **Hand Segmentation**: Subtract the background, apply thresholding, and find contours to segment the hand.
8. **Display Results**: Draw contours on the frame, display the thresholded image, and show the video feed.
9. **Save Images**: Save the thresholded hand image when the "s" key is pressed.
10. **Exit**: Release the camera and close all windows when the "q" key is pressed.

## References
- [OpenCV Documentation](https://docs.opencv.org/)
- [imutils Library](https://github.com/jrosebr1/imutils)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
