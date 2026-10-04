# Apple Sorting Task

import numpy as np
import random
import pandas as pd
import os
from psychopy.hardware.keyboard import Keyboard
from psychopy.gui import DlgFromDict
from psychopy import visual,core,data,logging
from get_task_design import TaskDesign
from helper_functions import *

# INITIALIZE VARIABLES
#exp_info = {'participant_number': ''}
#dlg = DlgFromDict(exp_info)
#participant_number = int(exp_info['participant_number'])
#participant_number=1
#filename = 'data/dat_'+str(participant_number)
participant_number = int(os.getenv('SUBJID'))
userprofile = os.getenv('userprofile')
filename = userprofile + '\\' + 'DATA\\dat_' + str(d.participant_number)

exp = data.ExperimentHandler(name='AppleSorting3',
                version='1.0',
                dataFileName=filename)

prac_d = TaskDesign(participant_number, practice=True)
d = TaskDesign(participant_number)
p_counter, b_counter = 0, 0

# CREATE WINDOW AND PRELOAD IMAGES/TEXT
win = visual.Window(fullscr=True, color=[1,1,1], monitor='Beeldscherm 1', units='deg')
win.setMouseVisible(False)
kb = Keyboard()
imgs = load_images(win)
fixation = imgs['fixation']
text = visual.TextStim(win, text='', color=[0,0,0], units='cm', height = 0.775)

skip_instructions = False
break_time = 30 # DO NOT FORGET TO CHANGE

# START INSTRUCTIONS AND PRACTICE BLOCKS
if not skip_instructions:
    for _ in range(2):
        # START INSTRUCTIONS
        show_instructions1(p_counter==0, prac_d.taskid[p_counter], win, kb, imgs)
        # FIRST PRACTICE
        practice_ok = False
        while not practice_ok:
            first_practice = data.TrialHandler(trialList = prac_d.stimList[p_counter], nReps = 1, name='practice', method='sequential')
            exp.addLoop(first_practice)
            # start of block screen
            text.text = 'Press space to start the practice block'
            text.draw()
            win.flip()
            kb.clearEvents()
            while True:
                keys = kb.getKeys()
                if 'space' in keys:
                    break
                elif 'z' in keys:
                    win.close()
                    core.quit()
            # if active task, first get button press
            if prac_d.taskid[p_counter]==1:
                resp_screen = imgs['resp_imgs'][0][prac_d.taskid[p_counter]][prac_d.epimap[p_counter]]
                resp_screen.draw()
                win.flip()
                kb.clearEvents()
                kb.clock.reset()
                while True:
                    keys = kb.getKeys(['e', 'p'])
                    if 'e' in keys:
                        prev_resp = 'e'
                        break
                    elif 'p' in keys:
                        prev_resp = 'p'
                        break
            else:
                prev_resp = '' 
            # initialize variables and loop over trials
            accuracy = []
            for thisTrial in first_practice:
                resp, correct, conf, rt = show_trial(thisTrial, exp, prev_resp, win, kb, imgs, False)
                prev_resp = resp
                if not correct==99:
                    accuracy.append(correct)
            # add one to p_counter
            if p_counter == 1:
                p_counter = 0
            elif p_counter == 5:
                p_counter = 4
            else:
                p_counter += 1
            # if finished?
            if np.mean(accuracy)>0.70:
                practice_ok = True
                if p_counter<2:
                    p_counter = 2
                else:
                    p_counter = 6
            else:
                text.text = 'It seems you do not fully understand the game yet. \n Press [R] for a recap of the instructions or press space to practice again'
                text.draw()
                win.flip()
                while True:
                    keys = kb.getKeys()
                    if 'space' in keys:
                        break
                    elif 'r' in keys:
                        recap = [visual.ImageStim(win, image='img/instr/instrC_recap.png', interpolate=True), visual.ImageStim(win, image='img/instr/instrO_recap.png', interpolate=True)][prac_d.taskid[p_counter]]
                        recap.draw()
                        win.flip()
                        while True:
                            keys = kb.getKeys()
                            if 'space' in keys:
                                break
                    if 'space' in keys:
                        break
        
                
        # CONTINUE INSTRUCTIONS
        show_instructions2(prac_d.taskid[p_counter], win, kb, imgs)
        # SECOND PRACTICE
        practice_ok = False
        while not practice_ok:
            # start of block screen
            text.text = 'Press space to start the practice block'
            text.draw()
            win.flip()
            kb.clearEvents()
            while True:
                keys = kb.getKeys()
                if 'space' in keys:
                    break
                elif 'z' in keys:
                    win.close()
                    core.quit()
            # create TrialHandler and add to experiment
            practice = data.TrialHandler(trialList = prac_d.stimList[p_counter], nReps=1, name='practice', method='sequential')
            exp.addLoop(practice)
            # if active task, first get button press
            if prac_d.taskid[p_counter]==1:
                resp_screen = imgs['resp_imgs'][1][prac_d.taskid[p_counter]][prac_d.epimap[p_counter]]
                resp_screen.draw()
                win.flip()
                kb.clearEvents()
                kb.clock.reset()
                while True:
                    keys = kb.getKeys(['e', 'r', 'o', 'p'])
                    if 'e' in keys:
                        prev_resp = 'e'
                        break
                    elif 'r' in keys:
                        prev_resp = 'r'
                        break
                    elif 'o' in keys:
                        prev_resp = 'o'
                        break
                    elif 'p' in keys:
                        prev_resp = 'p'
                        break
            else:
                prev_resp = '' 
            # initialize variables and loop over trials
            accuracy = []
            confidence = []
            for thisTrial in practice:
                resp, correct, conf, rt = show_trial(thisTrial, exp, prev_resp, win, kb, imgs)
                prev_resp = resp
                if not correct==99:
                    accuracy.append(correct)
                    confidence.append(conf)
            # add one to p_counter
            if p_counter == 3:
                p_counter = 2
            elif p_counter == 7:
                p_counter = 6
            else:
                p_counter += 1
            # if confidence too high or low
            if np.mean(confidence)<0.20 or np.mean(confidence)>0.80:
                text.text = 'Please remember to use the different buttons to indicate your confidence! \n Use the buttons [E] or [P] when you are RELATIVELY confident, or use the buttons [R] or [O] when you are RELATIVELY unsure. \n \n Press space to continue'
                text.draw()
                win.flip()
                while True:
                    keys = kb.getKeys()
                    if 'space' in keys:
                        break
            # if finished?
            if np.mean(accuracy)>0.60:
                practice_ok = True
                p_counter = 4
            else:
                text.text = 'It seems you do not fully understand the game yet. \n Press [R] for a recap of the instructions or press space to practice again'
                text.draw()
                win.flip()
                while True:
                    keys = kb.getKeys()
                    if 'space' in keys:
                        break
                    elif 'r' in keys:
                        recap = [visual.ImageStim(win, image='img/instr/instrC_recap.png', interpolate=True), visual.ImageStim(win, image='img/instr/instrO_recap.png', interpolate=True)][prac_d.taskid[p_counter]]
                        recap.draw()
                        win.flip()
                        while True:
                            keys = kb.getKeys()
                            if 'space' in keys:
                                break
                    if 'space' in keys:
                        break
                
text.text = 'Time for a short break! \n \n After the break, you will start with the main experiment. \n \n Before each block, you will be told whether you should perform the COMPUTER version or the ACTION version of the task.'
text.draw()
win.flip()
core.wait(break_time)

next_block = ['/8 \n \n Figure out from which tree the COMPUTER picks apples! \n \n \n Press space to continue',
'/8 \n \n Figure out which ACTION picks apples from the correct tree! \n \n \n Press space to continue']

# EXPERIMENTAL BLOCKS (30 SECOND BREAKS BETWEEN BLOCKS)
for _ in range(d.nBlocks):
    # save data
    exp.saveAsWideText(filename, fileCollisionMethod='overwrite')
    # start of block screen
    text.text = 'Block ' + str(b_counter+1) + next_block[d.taskid[b_counter]]
    text.draw()
    win.flip()
    kb.clearEvents()
    while True:
        keys = kb.getKeys()
        if 'space' in keys:
            break
        elif 'z' in keys:
            win.close()
            core.quit()
    # create TrialHandler and add to experiment
    main = data.TrialHandler(trialList = d.stimList[b_counter], nReps=1, name='main', method='sequential')
    exp.addLoop(main)
    # if active task, first get button press
    if d.taskid[b_counter]==1:
        resp_screen = imgs['resp_imgs'][1][d.taskid[b_counter]][d.epimap[b_counter]]
        resp_screen.draw()
        win.flip()
        kb.clearEvents()
        kb.clock.reset()
        while True:
            keys = kb.getKeys(['e', 'r', 'o', 'p'])
            if 'e' in keys:
                prev_resp = 'e'
                break
            elif 'r' in keys:
                prev_resp = 'r'
                break
            elif 'o' in keys:
                prev_resp = 'o'
                break
            elif 'p' in keys:
                prev_resp = 'p'
                break
    else:
        prev_resp = ''      
    # initialize variables and loop over trials
    accuracy = []
    confidence = []
    for thisTrial in main:
        resp, correct, conf, rt = show_trial(thisTrial, exp, prev_resp, win, kb, imgs)
        prev_resp = resp
        if not correct==99:
            accuracy.append(correct)
            confidence.append(conf)
    b_counter += 1
    # if confidence buttons not used
    if np.mean(confidence)<0.20 or np.mean(confidence)>0.80:
        text.text = 'Please remember to use the different buttons to indicate your confidence! \n Use the buttons [E] or [P] when you are RELATIVELY confident, or use the buttons [R] or [O] when you are RELATIVELY unsure. \n \n Press space to continue'
        text.draw()
        win.flip()
        while True:
            keys = kb.getKeys()
            if 'space' in keys:
                break
    # BREAK
    if not b_counter==(d.nBlocks):
        text.text = 'Time for a short break! \n \n Your performance in the last block was ' + str(int(np.mean(accuracy)*100)) + '%'
        text.draw()
        win.flip()
        if b_counter==((d.nBlocks//2)):
            core.wait(break_time*10)
        else:
            core.wait(break_time)

# END OF EXPERIMENT
text.text = 'This is the end of the experiment! \n \n Thank you very much for participating. \n \n Press space to end the experiment.'
text.draw()
win.flip()
kb.clearEvents()
while True:
    keys = kb.getKeys()
    if 'space' in keys:
        break
win.close()
core.quit()


