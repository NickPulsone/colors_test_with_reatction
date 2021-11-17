# colors_test_with_reatction
Implements a psychological test where a user is displayed a series of visual stimuli, each containing a word in colored test. The user must speak the color of the text (not the text itself) and the program will calculate the reaction time for each stimulus.

Requires Python 3.9. Edit tunable paramaters as necessary in "color_reaction.py."

IMPORTANT: Include the files in this drive link in your working directory (too big for github): 
https://drive.google.com/drive/folders/1_XCEDEXR9AgY9L-dRdYDVTmz9gXPXfcK?usp=sharing

If the program is unable to calculate the reaction time of a given response (whether it be the because the user failed to respond, the microphone did not pick up user audio, or otherwise) the reaction time will be recorded as "nan."  

For actual testing, use the programs in the "post_processing" branch. "run_colors.py" will run the test on a subject and save an audio file and csv file with the relevant data. "process_colors.py" will use this data to calculate reaction times, accuracies, etc. Post processing is mostly automatic, but does require review from a user to doule check the responses.
