# Real-time Sudoku Solver

This project is a real-time Sudoku solver that uses computer vision to detect a Sudoku puzzle from a live webcam feed, recognize the digits, solve the puzzle, and overlay the solution back onto the video stream.

*(Note: This project is a work in progress.)*

## Core Functionality

The application (`app.py`) operates in a continuous loop, processing frames from a webcam. The workflow for each frame is as follows:

1.  **Image Capture & Preprocessing**: The main script captures video frames using OpenCV. The image is then preprocessed (e.g., blurred, thresholded) to make it easier to detect the puzzle grid.
2.  **Grid Detection**: The system finds contours in the processed image. It searches for a large, four-cornered polygon, which it assumes to be the Sudoku grid (`current/process.py`).
3.  **Perspective Warping**: Once the four corners of the grid are identified, the image is warped to create a flat, top-down view of the puzzle.
4.  **Cell Extraction & Digit Recognition**:
    * The warped grid is processed to isolate the grid lines and create a mask, leaving only the numbers.
    * This image is then split into 81 individual squares (cells).
    * Each cell is cleaned and passed to a pre-trained TensorFlow/Keras CNN model (`digit_cnn.keras`) to recognize the digit it contains. Empty cells are identified as '0'.
5.  **Puzzle Solving**:
    * The recognized digits are formatted into a single string representing the 9x9 grid.
    * This string is passed to a solver in `current/sudoku.py`, which uses an exact cover algorithm to find the solution.
    * The main application caches solutions in a dictionary (`seen`) to avoid re-solving the same puzzle. If an impossible puzzle is detected, it is also cached and skipped in the future.
6.  **Display Solution**: If a solution is found, the solved digits are drawn onto the warped image in the corresponding empty cells. This solved grid is then unwarped and projected back onto the original video frame, overlaying the solution on the live puzzle.

The final output is displayed in a window titled 'Sudoku Solver - HaiShelby'.
