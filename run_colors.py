#!/usr/bin/env python
import ctypes
import datetime
from time import sleep
import cv2
import numpy as np
import sounddevice as sd
from scipy.io import loadmat, wavfile
import csv

""" ~~~~~~~~~~~~~     TUNABLE PARAMETERS     ~~~~~~~~~~~~~ """
# Name of given trial
TRIAL_NAME = "color_test"

# Number of stimuli
NUM_TESTS = 40

# Delay time between each visual stimulus
DELAY = 1.75

# Colors dictionary that identifies the RGB values of the used colors
COLORS = {"YELLOW": (0, 255, 255), "RED": (0, 0, 255), "GREEN": (0, 255, 0), "BLUE": (255, 0, 0), "BLACK": (0, 0, 0)}

# Name of the matlab file containing stimulus info (include filepath if necessary)
MAT_FILE_NAME = "ColorWord_versionB.mat"
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

# Get screen dimensions
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# Make sure cv2 images are displayed in full screen
window_name = 'projector'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, screensize[1] - 1, screensize[0] - 1)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Create a blank white image as a template
img = np.full((screensize[1], screensize[0], 3), fill_value=255, dtype=np.uint8)

# Define text parameters for stimuli images
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 10.0
fontThickness = 40
countDownFontScale = 7.0
coutDownFontThickness = 28

if __name__ == "__main__":
    # Open matfile, access colors for test, determine number of iterations/visuals
    mat = loadmat(MAT_FILE_NAME)
    color_words = [mat["words_test"][i].strip() for i in range(len(mat["words_test"]))]
    actual_colors = (255 * mat["colors_test"]).tolist()
    # iterations = len(color_words)
    iterations = NUM_TESTS

    # Convert Yellow from BGR in Matlab to RGB in Opencv
    for i in range(iterations):
        if actual_colors[i] == [255, 255, 0]:
            actual_colors[i] = list(reversed(actual_colors[i]))

    # Creates an array that contains the global time for each time stamp
    stimuli_time_stamps = np.empty(iterations, dtype=datetime.datetime)
    # Create an array of stimuli images
    stimuli_images = []
    for i in range(iterations):
        # Copy the template image
        new_img = np.copy(img)
        # Determine text size from given word
        textsize = cv2.getTextSize(color_words[i], font, 1, 2)[0]
        # Define parameters for positioning text on a given blank image
        textX = int((img.shape[1] - textsize[0] * fontScale) / 2)
        textY = int((img.shape[0] + textsize[1] * fontScale) / 2)
        bottomLeftCornerOfText = (textX, textY)
        # Position text on the screen
        cv2.putText(new_img, color_words[i],
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    color=actual_colors[i],
                    thickness=fontThickness)
        # Add the image to the array
        stimuli_images.append(new_img)
    # Give user a countdown
    for word in ["Get Ready...", "3..", "2..", "1..", "GO!!!"]:
        # Copy blank image from template
        new_img = np.copy(img)
        # Determine text size
        textsize = cv2.getTextSize(word, font, 1, 2)[0]
        # Define parameters for positioning text on template image
        textX = int((img.shape[1] - textsize[0] * countDownFontScale) / 2)
        textY = int((img.shape[0] + textsize[1] * countDownFontScale) / 2)
        bottomLeftCornerOfText = (textX, textY)
        # Position text on the screen
        cv2.putText(new_img, word,
                    bottomLeftCornerOfText,
                    font,
                    countDownFontScale,
                    color=(0, 0, 0),  # make the words black
                    thickness=coutDownFontThickness)
        # Wait out a 1s delay, then destory the image
        cv2.imshow(window_name, new_img)
        cv2.waitKey(1)
        sleep(1.0)
    sleep(0.5)

    # Define recording parameters and begin recording and start recording
    rec_seconds = int(iterations)*DELAY*1.2 + 5
    sample_rate = 44100
    myrecording = sd.rec(int(rec_seconds * sample_rate), samplerate=sample_rate, channels=1)
    recording_start_time = datetime.datetime.now()
    sleep(DELAY)

    # Displays the text to the user for given number of iterations
    for i in range(iterations):
        # Get global time of stimulus
        stimuli_time_stamps[i] = datetime.datetime.now()
        # Show image add the given array position to the user
        cv2.imshow(window_name, stimuli_images[i])
        # Wait out the given delay, then destory the image
        cv2.waitKey(1)
        sleep(DELAY)
    # Destroy last displayed image
    cv2.destroyAllWindows()
    # Stop the recording, save file as .wav
    print("Waiting for recording to stop...")
    sd.wait()
    wavfile.write(TRIAL_NAME + '.wav', sample_rate, myrecording)  # Save as WAV file
    print("Done.")

    # Calculate the time at which each stimulus is displayed with respect to the start of the recording
    stimuli_time_stamps = np.array(
        [(stimuli_time_stamps[i] - recording_start_time).total_seconds() for i in range(iterations)])

    # Init an array to hold the correct answers
    correct_answers = [list(COLORS.keys())[list(COLORS.values()).index(tuple(actual_colors[i]))] for i in
                       range(iterations)]

    # Write results to file
    with open(TRIAL_NAME + ".csv", 'w') as reac_file:
        writer = csv.writer(reac_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Text', 'Actual Color', 'Stimuli time from start (s)'])
        for i in range(iterations):
            writer.writerow([color_words[i], correct_answers[i], stimuli_time_stamps[i]])
    print("Done")
