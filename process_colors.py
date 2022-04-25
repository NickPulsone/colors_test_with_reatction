#!/usr/bin/env python
import numpy as np
import speech_recognition
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import csv
import speech_recognition as sr
import soundfile
from math import isnan
import os

""" ~~~~~~~~~~~~~     TUNABLE PARAMETERS     ~~~~~~~~~~~~~ """
# Name of given trial
TRIAL_NAME = "color_test"
CSV_FILENAME = TRIAL_NAME + ".csv"

# Delay time between each visual stimulus
DELAY = 1.75

# Colors dictionary that identifies the RGB values of the used colors
COLORS = {"YELLOW": (0, 255, 255), "RED": (0, 0, 255), "GREEN": (0, 255, 0), "BLUE": (255, 0, 0), "BLACK": (0, 0, 0)}

# Name of the matlab file containing stimulus info (include filepath if necessary)
MAT_FILE_NAME = "ColorWord_versionB.mat"

# The highest audio level (in dB) the program will determine to be considered "silence"
SILENCE_THRESHOLD_DB = -20.0

# The minimum period, in milliseconds, that could distinguish two different responses
MIN_PERIOD_SILENCE_MS = 100
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


# Normalize audio file to given target dB level - https://stackoverflow.com/questions/59102171/getting-timestamps-from-audio-using-pythons
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


if __name__ == "__main__":
    # Open csv file, get needed information
    file = open(CSV_FILENAME)
    reader = csv.reader(file)
    header = next(reader)
    data = []
    for row in reader:
        if len(row) > 0:
            data.append(row)
    data = np.array(data)
    color_words = np.array(data[:, 0], dtype=str)
    actual_colors = np.array(data[:, 1], dtype=str)
    stimuli_time_stamps = np.array(data[:, 2], dtype=float)
    NUM_TESTS = stimuli_time_stamps.size

    print("Interpret data (this may take a while)...")
    # Open .wav with pydub
    audio_segment = AudioSegment.from_wav(TRIAL_NAME + ".wav")
    rec_seconds = audio_segment.duration_seconds

    # Normalize audio_segment to a threshold
    normalized_sound = match_target_amplitude(audio_segment, SILENCE_THRESHOLD_DB)

    # Generate nonsilent chunks (start, end) with pydub
    response_timing_chunks = np.array(
        detect_nonsilent(normalized_sound, min_silence_len=MIN_PERIOD_SILENCE_MS, silence_thresh=SILENCE_THRESHOLD_DB,
                         seek_step=1))

    # If unable to detect nonsilence, end program and notify user
    if len(response_timing_chunks) == 0:
        print("Could not detect user's responses. Silence threshold/Minimum silence period may need tuning.")
        exit(1)

    # Store the time that the user starts to speak in each nonsilent "chunk"
    response_timing_markers = np.array(response_timing_chunks[:, 0]) / 1000.0

    # Create a folder to store the individual responses as clips to help determine
    # response accuracies later on.
    clip_seperation_path = TRIAL_NAME + "_reponse_chunks"
    if not os.path.isdir(clip_seperation_path):
        os.mkdir(clip_seperation_path)
    # How much we add (ms) to the ends of a clip when saved
    clip_threshold = 600
    for i in range(len(response_timing_chunks)):
        chunk = response_timing_chunks[i]
        chunk_filename = os.path.join(clip_seperation_path, f"chunk{i}.wav")
        # Save the chunk as a serperate wav, acounting for the fact it could be at the very beggining or end
        if chunk[0] <= clip_threshold:
            (audio_segment[0:chunk[1]+clip_threshold]).export(chunk_filename, format="wav")
        elif chunk[1] >= ((rec_seconds*1000.0) - clip_threshold - 1):
            (audio_segment[chunk[0]-clip_threshold:(rec_seconds*1000)-1]).export(chunk_filename, format="wav")
        else:
            (audio_segment[chunk[0]-clip_threshold:chunk[1]+clip_threshold]).export(chunk_filename, format="wav")
        # Reformat the wav files using soundfile to allow for speech recongition, and store in folder
        data, samplerate = soundfile.read(chunk_filename)
        soundfile.write(chunk_filename, data, samplerate, subtype='PCM_16')

    # Init an array to hold the users raw response
    raw_answers = []
    # Init an array to hold the accuracy of the user's response (TRUE, FALSE, or N/A)
    response_accuracies = []

    # Init the speech to text recognizer
    r = sr.Recognizer()

    # Calculate the reponse times given the arrays for response_timing_markers and stimuli_time_stamps
    reaction_times = []
    # Init an array to contain the indices of the nonsilent clips used
    clip_index_array = np.empty(NUM_TESTS, dtype=int)
    # Keep track of total correct answers to track user performance
    num_correct_responses = 0
    for i in range(NUM_TESTS):
        # If there is no response after a time stamp, clearly the user failed to respond...
        clip_index_array[i] = -9999
        rt = float('nan')
        if stimuli_time_stamps[i] > response_timing_markers[-1]:
            response_accuracies.append("N/A")
            raw_answers.append("N/A")
        # ..otherwise go through more tests
        else:
            # Determine the most accurate nonsilent chunk that is associated with a given iteration
            for j in range(len(response_timing_markers)):
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
                reaction_times.append(float('nan'))
                raw_answers.append("N/A")
                response_accuracies.append("N/A")
                continue
            else:
                # If the response was valid, detemine if it was correct using speech recognition
                with sr.AudioFile(os.path.join(clip_seperation_path, f"chunk{j}.wav")) as source:
                    clip_index_array[i] = j
                    # listen for the data (load audio to memory)
                    audio_data = r.record(source)
                    # recognize (convert from speech to text)
                    try:
                        resp = (r.recognize_google(audio_data).split()[0]).upper()
                        resp_backup = (r.recognize_sphinx(audio_data).split()[0]).upper()
                    # If no response can be determined, report accuracies as N/A, store reaction time, and move on
                    except speech_recognition.UnknownValueError as err:
                        response_accuracies.append("N/A")
                        raw_answers.append("N/A")
                        reaction_times.append(rt)
                        continue
                    # compare response from stt to the acutal response, update response accuracies accordingly
                    if resp in COLORS.keys():
                        if resp == actual_colors[i]:
                            response_accuracies.append("TRUE")
                            num_correct_responses += 1
                        else:
                            response_accuracies.append("FALSE")
                        raw_answers.append(resp)
                    elif resp_backup in COLORS.keys():
                        if resp_backup == actual_colors[i]:
                            response_accuracies.append("TRUE")
                            num_correct_responses += 1
                        else:
                            response_accuracies.append("FALSE")
                        raw_answers.append(resp_backup)
                    # If word not found, store response and mark as false
                    else:
                        raw_answers.append(resp)
                        response_accuracies.append("FALSE")
        reaction_times.append(rt)

    # Create another array to label each reaction time according to if it was within the allotted time or not
    reaction_on_time = np.empty(NUM_TESTS, dtype=bool)
    for i in range(NUM_TESTS):
        if reaction_times[i] > DELAY or isnan(reaction_times[i]):
            reaction_on_time[i] = False
        else:
            reaction_on_time[i] = True

    # Write results to file
    with open(TRIAL_NAME + "_RESULTS.csv", 'w') as reac_file:
        writer = csv.writer(reac_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['Text', 'Actual Color', 'Response', 'Accuracy (T/F)',
                'Reaction time (s)', 'Reaction on time (T/F)', 'Clip Index', ' ', ' ', 'Time (from start) user speaks'])
        num_rows_in_table = max([len(response_timing_markers), len(actual_colors)])
        for i in range(num_rows_in_table):
            if i >= len(response_timing_markers):
                writer.writerow([color_words[i], actual_colors[i], raw_answers[i], response_accuracies[i], reaction_times[i],
                                reaction_on_time[i], clip_index_array[i], ' ', ' ', -1])
            elif i >= len(actual_colors):
                writer.writerow(
                    [-1, -1, -1, -1, -1, -1, -1, ' ', ' ', response_timing_markers[i]])
            else:
                writer.writerow(
                    [color_words[i], actual_colors[i], raw_answers[i], response_accuracies[i], reaction_times[i],
                     reaction_on_time[i], clip_index_array[i], ' ', ' ', response_timing_markers[i]])
    print("Done")
