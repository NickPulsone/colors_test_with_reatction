#!/usr/bin/env python
import ctypes
import datetime
from time import sleep
import cv2
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.io import loadmat, wavfile
import csv
import speech_recognition as sr
import soundfile
from math import isnan

""" ~~~~~~~~~~~~~     TUNABLE PARAMETERS     ~~~~~~~~~~~~~ """
# Name of given trial
TRIAL_NAME = "test1"

# Delay time between each visual stimulus
DELAY = 1.0

# Colors dictionary that identifies the RGB values of the used colors
COLORS = {"YELLOW": (0, 255, 255), "RED": (0, 0, 255), "GREEN": (0, 255, 0), "BLUE": (255, 0, 0), "BLACK": (0, 0, 0)}

# Name of the matlab file containing stimulus info (include filepath if necessary)
MAT_FILE_NAME = "ColorWord_versionB.mat"

# The highest audio level (in dB) the program will determine to be considered "silence"
SILENCE_THRESHOLD_DB = -20.0

# The minimum period, in milliseconds, that could distinguish two different responses
MIN_PERIOD_SILENCE_MS = 250

# If you already have an audio file (.wav) with the proper name in the working directory, set to True
# SKIP_RECORDING = True
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


# Normalize audio file to given target dB level - https://stackoverflow.com/questions/59102171/getting-timestamps-from-audio-using-pythons
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


if __name__ == "__main__":
    # Open matfile, access colors for test, determine number of iterations/visuals
    mat = loadmat(MAT_FILE_NAME)
    color_words = [mat["words_test"][i].strip() for i in range(len(mat["words_test"]))]
    actual_colors = (255 * mat["colors_test"]).tolist()
    iterations = int(len(color_words) / 15)

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
    rec_seconds = int(iterations) + 10
    sample_rate = 44100
    myrecording = sd.rec(int(rec_seconds * sample_rate), samplerate=sample_rate, channels=1)
    recording_start_time = datetime.datetime.now()
    # Displays the text to the user for given number of iterations
    for i in range(iterations):
        # Show image add the given array position to the user
        cv2.imshow(window_name, stimuli_images[i])
        # Get global time of stimulus
        stimuli_time_stamps[i] = datetime.datetime.now()
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
    print("Calculating reaction times...")
    # Calculate the time at which each stimulus is displayed with respect to the start of the recording
    stimuli_time_stamps = np.array([(stimuli_time_stamps[i] - recording_start_time).total_seconds() for i in range(iterations)])
    # Open .wav with pydub
    audio_segment = AudioSegment.from_wav(TRIAL_NAME + ".wav")
    # Normalize audio_segment to a threshold
    normalized_sound = match_target_amplitude(audio_segment, SILENCE_THRESHOLD_DB)
    # Generate nonsilent chunks (start, end) with pydub
    response_timing_chunks = np.array(detect_nonsilent(normalized_sound, min_silence_len=MIN_PERIOD_SILENCE_MS, silence_thresh=SILENCE_THRESHOLD_DB, seek_step=1))
    # If unable to detect nonsilence, end program and notify user
    if len(response_timing_chunks) == 0:
        print("Could not detect user's responses. Silence threshold/Minimum silence period may need tuning.")
        exit(1)
    # Calculate the time that the user starts to speak in each nonsilent "chunk"
    response_timing_markers = np.array(response_timing_chunks[:, 0]) / 1000.0
    # Calculate the reponse times given the arrays for response_timing_markers and stimuli_time_stamps
    reaction_times = []
    for i in range(iterations):
        # Determine the most accurate nonsilent chunk that is associated with a given iteration
        for j in range(len(response_timing_markers)):
            if response_timing_markers[j] > stimuli_time_stamps[i]:
                # If reaction is too fast, it means the program is considering a delayed response from previous stimulus
                # Thus, we should continue the loop if that is the case, otherwise, break and store the reaction time
                if response_timing_markers[j] - stimuli_time_stamps[i] < 0.2 and reaction_times[-1] > 1.0:
                    continue
                rt = response_timing_markers[j] - stimuli_time_stamps[i]
                break
        # If there is no nonsilent chunk after the time that the stimulus is displayed, store reaction time as "nan"
        # Also if the user's response is over 1.2s after the stimulus is displayed, then we know they either failed to
        # respond or the audio was not recorded and intepreted properly.
        if j >= len(response_timing_markers) or rt > 1.2:
            rt = float('nan')
        reaction_times.append(rt)


    # Initialize an array containing the correct answers, and another array to hold user response accuracies
    correct_answers = [list(COLORS.keys())[list(COLORS.values()).index(tuple(actual_colors[i]))] for i in range(iterations)]
    response_accuracies = np.empty(iterations, dtype=str)

    # Convert the .wav to PCIM Wav that can be read by the speech recognizer
    data, samplerate = soundfile.read(TRIAL_NAME + ".wav")
    soundfile.write(TRIAL_NAME + ".wav", data, samplerate, subtype='PCM_16')

    # Start the google speech to text recognizer
    r = sr.Recognizer()
    # Open the .wav file
    with sr.AudioFile(TRIAL_NAME + ".wav") as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
    # Get individual answers
    answers = (text.upper()).split()

    # Keep track of total correct answers to track user performance
    num_correct_responses = 0
    # Ensure that if a response was miss, it is not cross-compared with the true correct answers
    for i in range(iterations):
        if isnan(reaction_times[i]):
            print("GOT HERE")
            answers.insert(i, "N/A")
    # Determine if the response was correct, if not, store what the speech recognition thought
    for i in range(iterations):
        if answers[i] == "N/A":
            response_accuracies[i] = "N/A"
        else:
            try:
                if answers[i][0] == correct_answers[i][0]:
                    response_accuracies[i] = "TRUE"
                    num_correct_responses += 1
                else:
                    response_accuracies[i] = "FALSE"
            # Index error means that there was a recording issue, so just assign "N/A"
            except IndexError as err:
                response_accuracies[i] = "N/A"

    print("You got " + str(num_correct_responses) + " / " + str(iterations) +
          " correct answers (" + str(100*float(num_correct_responses)/iterations) + " %).")
    # Write results to file
    with open(TRIAL_NAME + ".csv", 'w') as reac_file:
        writer = csv.writer(reac_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Text', 'Actual Color', 'Response', 'Accuracy (T/F)', 'Reaction time (s)'])
        for i in range(iterations):
            writer.writerow([color_words[i], correct_answers[i], answers[i], response_accuracies[i], reaction_times[i]])
    print("Done")

