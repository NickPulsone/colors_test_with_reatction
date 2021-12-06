#!/usr/bin/env python
import numpy as np
import csv
import speech_recognition as sr
import os
from math import isnan

# Clip indices of clips to discard
REMOVE_CLIPS = []

# Colors dictionary that identifies the RGB values of the used colors
COLORS = {"YELLOW": (0, 255, 255), "RED": (0, 0, 255), "GREEN": (0, 255, 0), "BLUE": (255, 0, 0), "BLACK": (0, 0, 0)}

# Pause time in seconds
DELAY = 1.0

# Trial name and name of csv file containing existing results to be modified
TRIAL_NAME = "color_test1"
TRIAL_CSV_FILENAME = TRIAL_NAME + ".csv"
RESULTS_CSV_FILENAME = TRIAL_NAME + "_results.csv"
CLIP_SEPERATION_PATH = TRIAL_NAME + "_reponse_chunks"
CHUNK_DIR_NAME = "color_test1_reponse_chunks"

# Get data from trial csv file
trial_file = open(TRIAL_CSV_FILENAME)
trial_reader = csv.reader(trial_file)
trial_header = next(trial_reader)
data = []
for row in trial_reader:
    if len(row) > 0:
        data.append(row)
data = np.array(data)

# Extract necessary data from trial csv file
stimuli_time_stamps = np.array(data[:, 2], dtype=float)

# Get data from results csv file
results_file = open(RESULTS_CSV_FILENAME)
results_reader = csv.reader(results_file)
results_header = next(results_reader)
data = []
for row in results_reader:
    if len(row) > 0:
        data.append(row)
data = np.array(data)

# Extract necessary data from results csv file
color_words = np.array(data[:, 0], dtype=str)
actual_colors = np.array(data[:, 1], dtype=str)
user_responses = np.array(data[:, 2], dtype=str)
accuracy_array = np.array(data[:, 3], dtype=str)
reaction_times = np.array(data[:, 4], dtype=float)
reaction_on_time = np.array(data[:, 5], dtype=str)
clip_index_array = np.array(data[:, 6], dtype=int)
response_timing_markers = np.array(data[:, 9], dtype=float)

# Reconstruct arrays from formatted data in the sheet
color_words = color_words[color_words != '-1']
actual_colors = actual_colors[actual_colors != '-1']
user_responses = user_responses[user_responses != '-1']
accuracy_array = accuracy_array[accuracy_array != '-1']
reaction_times = reaction_times[reaction_times != -1]
reaction_on_time = reaction_on_time[reaction_on_time != '-1']
clip_index_array = clip_index_array[clip_index_array != -1]
response_timing_markers = response_timing_markers[response_timing_markers != -1]
NUM_TESTS = actual_colors.size

# Get the number of clips by counting the nubmer of clips in the folder
total_num_clips = 0
dir = CHUNK_DIR_NAME
for path in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, path)):
        total_num_clips += 1

# Get index of the iteration of each corresponding clip in question
num_remove_clips = len(REMOVE_CLIPS)
iteration_indices = np.empty(num_remove_clips, dtype=int)
for i in range(num_remove_clips):
    iteration_indices[i] = np.where(clip_index_array == REMOVE_CLIPS[i])[0][0]
clip_iteration_range = tuple(i for i in range(total_num_clips) if i not in REMOVE_CLIPS)

# Init the speech to text recognizer
r = sr.Recognizer()
for i in iteration_indices:
    # If there is no response after a time stamp, clearly the user failed to respond...
    clip_index_array[i] = -9999
    rt = float('nan')
    if stimuli_time_stamps[i] > response_timing_markers[-1]:
        accuracy_array[i] = "N/A"
        user_responses[i] = "N/A"
    # ..otherwise go through more tests
    else:
        # Determine the most accurate nonsilent chunk that is associated with a given iteration
        for j in clip_iteration_range:
            if response_timing_markers[j] > stimuli_time_stamps[i]:
                # If reaction is too fast, it means the program is considering a delayed response from previous stimulus
                # Thus, we should continue the loop if that is the case, otherwise, break and store the reaction time
                if response_timing_markers[j] - stimuli_time_stamps[i] < 0.1 and reaction_times[-1] > DELAY:
                    continue
                rt = response_timing_markers[j] - stimuli_time_stamps[i]
                # Break from the loop as soon as we find a response after the time of the stimulus
                break
        # If there is no nonsilent chunk after the time that the stimulus is displayed, store reaction time as "nan"
        if j >= len(response_timing_markers) or (rt > DELAY * 1.2):
            reaction_times[i] = float('nan')
            user_responses[i] = "N/A"
            accuracy_array[i] = "N/A"
            continue
        else:
            # If the response was valid, detemine if it was correct using speech recognition
            with sr.AudioFile(os.path.join(CLIP_SEPERATION_PATH, f"chunk{j}.wav")) as source:
                clip_index_array[i] = j
                # listen for the data (load audio to memory)
                audio_data = r.record(source)
                # recognize (convert from speech to text)
                try:
                    resp = (r.recognize_google(audio_data).split()[0]).upper()
                    resp_backup = (r.recognize_sphinx(audio_data).split()[0]).upper()
                # If no response can be determined, report accuracies as N/A, store reaction time, and move on
                except sr.UnknownValueError as err:
                    accuracy_array[i] = "N/A"
                    user_responses[i] = "N/A"
                    reaction_times[i] = rt
                    continue
                # compare response from stt to the acutal response, update response accuracies accordingly
                if resp in COLORS.keys():
                    if resp == actual_colors[i]:
                        accuracy_array[i] = "TRUE"
                    else:
                        accuracy_array[i] = "FALSE"
                    user_responses[i] = resp
                elif resp_backup in COLORS.keys():
                    if resp_backup == actual_colors[i]:
                        accuracy_array[i] = "TRUE"
                    else:
                        accuracy_array[i] = "FALSE"
                    user_responses[i] = resp_backup
                # If word not found, store response and mark as false
                else:
                    user_responses[i] = resp
                    accuracy_array[i] = "FALSE"
    reaction_times[i] = rt

# Create another array to label each reaction time according to if it was within the allotted time or not
reaction_on_time = np.empty(NUM_TESTS, dtype=bool)
for i in iteration_indices:
    if reaction_times[i] > DELAY or isnan(reaction_times[i]):
        reaction_on_time[i] = False
    else:
        reaction_on_time[i] = True

# Write results to file
with open(TRIAL_NAME + "_RESULTS.csv", 'w') as reac_file:
    writer = csv.writer(reac_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        ['Text', 'Actual Color', 'Response', 'Accuracy (T/F)', 'Reaction time (s)', 'Reaction on time (T/F)',
         'Clip Index', ' ', ' ', 'Time (from start) user speaks'])
    num_rows_in_table = max([len(response_timing_markers), len(actual_colors)])
    for i in range(num_rows_in_table):
        if i >= len(response_timing_markers):
            writer.writerow(
                [color_words[i], actual_colors[i], user_responses[i], accuracy_array[i], reaction_times[i],
                 reaction_on_time[i], clip_index_array[i], ' ', ' ', -1])
        elif i >= len(actual_colors):
            writer.writerow(
                [-1, -1, -1, -1, -1, -1, -1, ' ', ' ', response_timing_markers[i]])
        else:
            writer.writerow(
                [color_words[i], actual_colors[i], user_responses[i], accuracy_array[i], reaction_times[i],
                 reaction_on_time[i], clip_index_array[i], ' ', ' ', response_timing_markers[i]])
print("Done")
