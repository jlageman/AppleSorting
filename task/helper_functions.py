import numpy as np
import random
import pandas as pd
import os
from psychopy.hardware.keyboard import Keyboard
from psychopy import visual,core,data,logging
from scipy.stats import truncexpon, vonmises

## LOAD IMAGES ##

def load_images(win, Asc=3):
    sz = [Asc,Asc*(305/285)]
    posx = [(-Asc//2 - Asc), (-Asc//2), (Asc//2), (Asc//2+Asc)]
    # general images: gen_imgs[0-10]
    # 0 = wrong, 1 = repeat instruction, 2 = start practice, 3 = end of experiment, 4 = next block, 5 = break
    # 6 = next_practice, 7 = next block [0 = C and 1 = O], 8 = end of practice, 9 = second practice, 10 = respond faster
    imgs = {'fixation': visual.ImageStim(win, image='img/fixation.png', interpolate=True),\
    # response images: resp_imgs[demo][taskid][epimap]
    'resp_imgs':\
    [[[visual.ImageStim(win,image='img/BlueOrangeDemo.png'), visual.ImageStim(win,image='img/OrangeBlueDemo.png')],\
    [visual.ImageStim(win,image='img/BlueDemo.png'), visual.ImageStim(win,image='img/OrangeDemo.png')]],\
    [[visual.ImageStim(win,image='img/BlueOrange.png'), visual.ImageStim(win,image='img/OrangeBlue.png')],\
    [visual.ImageStim(win,image='img/BlueBasket.png'), visual.ImageStim(win,image='img/OrangeBasket.png')]]],\
    # apple images: apple_imgs[evd-level][color][position] (1,2,3,4 ; blue=0, orange=1; left1, left2, right1, right2)
    'apple_imgs':\
    [[[visual.ImageStim(win,image='img/1.png', size=sz, pos=(x,0)) for x in posx],\
    [visual.ImageStim(win,image='img/-1.png', size=sz, pos=(x,0)) for x in posx]],\
    [[visual.ImageStim(win,image='img/2.png', size=sz, pos=(x,0)) for x in posx],\
    [visual.ImageStim(win,image='img/-2.png', size=sz, pos=(x,0)) for x in posx]],\
    [[visual.ImageStim(win,image='img/3.png', size=sz, pos=(x,0)) for x in posx],\
    [visual.ImageStim(win,image='img/-3.png', size=sz, pos=(x,0)) for x in posx]],\
    [[visual.ImageStim(win,image='img/4.png', size=sz, pos=(x,0)) for x in posx],\
    [visual.ImageStim(win,image='img/-4.png', size=sz, pos=(x,0)) for x in posx]]]}
    # return all lists
    return imgs


## INSTRUCTION FUNCTIONS ##
    
def show_instructions1(first_time, taskid, win, kb, imgs):
    # initialize images
    fixation = imgs['fixation']
    press_button = visual.ImageStim(win, image='img/instr/press_button.png')
    
    # load task-specific images
    if taskid==0:
        task_imgs = [[visual.ImageStim(win, image='img/instr/instrC1A.png', interpolate=True), visual.ImageStim(win, image='img/instr/instrC1B.png', interpolate=True)],\
        visual.ImageStim(win, image='img/instr/instrC2.png', interpolate=True),\
        visual.ImageStim(win, image='img/instr/instrC3.png', interpolate=True),\
        visual.ImageStim(win, image='img/instr/instrC4.png', interpolate=True),\
        visual.ImageStim(win, image='img/instr/instrC5.png', interpolate=True)]
    elif taskid==1:
        task_imgs = [[visual.ImageStim(win, image='img/instr/instrO1A.png', interpolate=True), visual.ImageStim(win, image='img/instr/instrO1B.png', interpolate=True)],\
        visual.ImageStim(win, image='img/instr/instrO2.png', interpolate=True),\
        [visual.ImageStim(win, image='img/instr/instrO3A.png', interpolate=True), visual.ImageStim(win, image='img/instr/instrO3B.png', interpolate=True)],\
        visual.ImageStim(win, image='img/instr/instrO4.png', interpolate=True),\
        visual.ImageStim(win, image='img/instr/instrO5.png', interpolate=True),\
        visual.ImageStim(win, image='img/instr/instrO6.png', interpolate=True),\
        visual.ImageStim(win, image='img/instr/instrO7.png', interpolate=True)]
    
    # if first time, show general instructions first
    if first_time:
        gen_imgs = [visual.ImageStim(win, image='img/instr/instr1.png', interpolate=True), visual.ImageStim(win, image='img/instr/instr2.png', interpolate=True),\
        visual.ImageStim(win, image='img/instr/instr3.png', interpolate=True), visual.ImageStim(win, image='img/instr/instr4.png', interpolate=True)]
        for img in gen_imgs:
            img.draw()
            win.flip()
            core.wait(0.1)
            kb.clearEvents()
            while True:
                keys = kb.getKeys()
                if 'space' in keys:
                    break

    # show instructions
    for num, img in enumerate(task_imgs):

        # PASSIVE
        if taskid==0:
            # show image
            if num==0:
                img[first_time==0].draw()
            else:
                img.draw()
            win.flip()
            core.wait(0.2)
            
            # collect response
            if num==1:
                kb.clearEvents()
                while True:
                    keys = kb.getKeys()
                    if 'space' in keys:
                        break
                # show apples
                stim_list = [[imgs['apple_imgs'][2][0][0], imgs['apple_imgs'][1][0][1], imgs['apple_imgs'][1][1][2], imgs['apple_imgs'][3][0][3]],\
                [imgs['apple_imgs'][3][0][0], imgs['apple_imgs'][0][1][1], imgs['apple_imgs'][1][0][2], imgs['apple_imgs'][2][0][3]],\
                [imgs['apple_imgs'][2][1][0], imgs['apple_imgs'][2][1][1], imgs['apple_imgs'][1][0][2], imgs['apple_imgs'][3][1][3]]]
                for stims in stim_list:
                    fixation.draw()
                    win.flip()
                    core.wait(2)
                    # draw stimuli
                    for stim in stims:
                        stim.draw()
                    win.flip()
                    core.wait(1)
                fixation.draw()
                win.flip()
                core.wait(2)
                
            elif num==2:
                kb.clearEvents()
                while True:
                    keys = kb.getKeys()
                    if 'p' in keys:
                        feedback = visual.ImageStim(win, image='img/instr/correct.png')
                        feedback.draw()
                        win.flip()
                        core.wait(1)
                        break
                    elif 'e' in keys:
                        feedback = visual.ImageStim(win, image='img/instr/wrong.png')
                        feedback.draw()
                        win.flip()
                        core.wait(1)
                        break
            else:
                kb.clearEvents()
                while True:
                    keys = kb.getKeys()
                    if 'space' in keys:
                        break
    
        # ACTIVE
        elif taskid==1:
            # show image
            if num==0:
                img[first_time==0].draw()
            elif num==2:
                img[right_button_pressed].draw()
            else:
                img.draw()
            win.flip()
            core.wait(0.2)
            # collect response
            if num==1:
                kb.clearEvents()
                while True:
                    keys = kb.getKeys()
                    if 'space' in keys:
                        break
                # participant presses button
                press_button.draw()
                win.flip()
                kb.clearEvents()
                while True:
                    keys = kb.getKeys(['e', 'p'])
                    if 'e' in keys:
                        right_button_pressed = False
                        break
                    elif 'p' in keys:
                        right_button_pressed = True
                        break
                # show fixation
                fixation.draw()
                win.flip()
                core.wait(0.5)
                # show apples
                stims = [imgs['apple_imgs'][3][0][0], imgs['apple_imgs'][0][1][1], imgs['apple_imgs'][1][0][2], imgs['apple_imgs'][2][0][3]]
                for stim in stims:
                    stim.draw()
                win.flip()
                core.wait(1)
                # show fixation
                fixation.draw()
                win.flip()
                core.wait(1)
                
            elif num==2:
                kb.clearEvents()
                while True:
                    keys = kb.getKeys()
                    if 'y' in keys:
                        feedback = visual.ImageStim(win, image='img/instr/correct.png', interpolate=True)
                        feedback.draw()
                        win.flip()
                        core.wait(1)
                        break
                    elif 'e' in keys or 'p' in keys:
                        feedback = visual.ImageStim(win, image='img/instr/wrong.png', interpolate=True)
                        feedback.draw()
                        win.flip()
                        core.wait(1)
                        break
                        
            elif num==3:
                kb.clearEvents()
                while True:
                    keys = kb.getKeys()
                    if 'space' in keys:
                        break
                # start with orange
                col = 1
                for i in range(3):
                    # participant presses button
                    press_button.draw()
                    win.flip()
                    kb.clearEvents()
                    while True:
                        keys = kb.getKeys(['e', 'p'])
                        if 'e' in keys:
                            if i<1:
                                prev_ans = 'e'
                            else:
                                ans = 'e'
                            break
                        elif 'p' in keys:
                            if i<1:
                                prev_ans = 'p'
                            else:
                                ans = 'p'
                            break
                    # show fixation
                    fixation.draw()
                    win.flip()
                    core.wait(0.5)
                    # show apples
                    if i>=1:
                        if ans==prev_ans:
                            col = 0
                        else:
                            col = 1
                        prev_ans = ans
                    stims = [imgs['apple_imgs'][np.random.choice([2,3])][col][0], imgs['apple_imgs'][np.random.choice([0,1,2])][col][1], imgs['apple_imgs'][np.random.choice([0,1])][int(col==0)][2], imgs['apple_imgs'][np.random.choice([1,2,3])][col][3]]
                    for stim in stims:
                        stim.draw()
                    win.flip()
                    core.wait(1)
                    # show fixation
                    fixation.draw()
                    win.flip()
                    core.wait(1)
                            
            elif num==4:
                kb.clearEvents()
                while True:
                    keys = kb.getKeys()
                    if 'e' in keys:
                        feedback = visual.ImageStim(win, image='img/instr/correct.png', interpolate=True)
                        feedback.draw()
                        win.flip()
                        core.wait(1)
                        break
                    elif 'p' in keys:
                        feedback = visual.ImageStim(win, image='img/instr/wrong.png', interpolate=True)
                        feedback.draw()
                        win.flip()
                        core.wait(1)
                        break
                        
            else:
                kb.clearEvents()
                while True:
                    keys = kb.getKeys()
                    if 'space' in keys:
                        break

def show_instructions2(taskid, win, kb, imgs):
    fixation = imgs['fixation']
    # load task-specific images
    if taskid==0:
        task_imgs = [visual.ImageStim(win, image='img/instr/instrC6.png', interpolate=True),\
        visual.ImageStim(win, image='img/instr/instrC7.png', interpolate=True)]
    elif taskid==1:
        task_imgs = [visual.ImageStim(win, image='img/instr/instrO8.png', interpolate=True),\
        visual.ImageStim(win, image='img/instr/instrO9.png', interpolate=True)]
    # show instructions
    for img in task_imgs:
        # show image
        img.draw()
        win.flip()
        core.wait(0.2)
        # collect response
        kb.clearEvents()
        while True:
            keys = kb.getKeys()
            if 'space' in keys:
                break


## MAIN EXPERIMENT ##
    
def show_trial(trial, exp, prev_resp, win, kb, imgs, use_confidence=True):
    fixation = imgs['fixation']
    resp_screen = imgs['resp_imgs'][use_confidence][trial['taskid']][trial['epimap']]
    
    ## CUE TASK
    if trial['taskid']==0:
        
        ## PRESENT STIMULUS
        order = [0,1,2,3]
        stims = [imgs['apple_imgs'][int(abs(s)-1)][int(np.clip(np.sign(s),0,1)==trial['seqdir'])][o] for s,o in zip(trial['seqlvl'], order)]
        # draw fixation
        fixation.draw()
        win.flip()
        core.wait(0.5)
        # draw stimuli
        for stim in stims:
            stim.draw()
        win.flip()
        core.wait(1)
        # draw fixation
        fixation.draw()
        win.flip()
        core.wait(1)
        
        ## GET RESPONSE
        kb.clearEvents()
        kb.clock.reset()
        fixation.draw()
        resp_screen.draw()
        win.flip()
        while True:
            keys = kb.getKeys(['e', 'r', 'o', 'p'])
            if trial['correct_HC'] in keys:
                resp = trial['correct_HC']
                correct = 1
                conf = 1
                for key in keys:
                    rt = round(key.rt, 4)
                break
            elif trial['correct_LC'] in keys:
                resp = trial['correct_LC']
                correct = 1
                conf = 0
                for key in keys:
                    rt = round(key.rt, 4)
                break
            elif trial['incorrect_HC'] in keys:
                resp = trial['incorrect_HC']
                correct = 0
                conf = 1
                for key in keys:
                    rt = round(key.rt, 4)
                break
            elif trial['incorrect_LC'] in keys:
                resp = trial['incorrect_LC']
                correct = 0
                conf = 0
                for key in keys:
                    rt = round(key.rt, 4)
                break
            
    ## ACTION TASK 
    if trial['taskid']==1:
        
        ## PRESENT STIMULUS
        order = [2,3,0,1]
        stims = [imgs['apple_imgs'][int(abs(s)-1)][int(int(np.clip(np.sign(s),0,1)==trial['seqdir'])!=['e','p','r','o'].index(prev_resp)%2)][o] for s,o in zip(trial['seqlvl'], order)]
        # draw fixation
        fixation.draw()
        win.flip()
        core.wait(0.5)
        # draw stimuli
        for stim in stims:
            stim.draw()
        win.flip()
        core.wait(1)
        # draw fixation
        fixation.draw()
        win.flip()
        core.wait(1)
            
        ## GET RESPONSE
        if not trial['trialnum']==trial['nTrials']:
            fixation.draw()
            resp_screen.draw()
            win.flip()
            kb.clearEvents()
            kb.clock.reset()
            while True:
                keys = kb.getKeys(['e', 'r', 'o', 'p'])
                if trial['correct_HC'] in keys:
                    resp = trial['correct_HC']
                    correct = 1
                    conf = 1
                    for key in keys:
                        rt = round(key.rt, 4)
                    break
                elif trial['correct_LC'] in keys:
                    resp = trial['correct_LC']
                    correct = 1
                    conf = 0
                    for key in keys:
                        rt = round(key.rt, 4)
                    break
                elif trial['incorrect_HC'] in keys:
                    resp = trial['incorrect_HC']
                    correct = 0
                    conf = 1
                    for key in keys:
                        rt = round(key.rt, 4)
                    break
                elif trial['incorrect_LC'] in keys:
                    resp = trial['incorrect_LC']
                    correct = 0
                    conf = 0
                    for key in keys:
                        rt = round(key.rt, 4)
                    break
        else:
            resp = 99
            correct = 99
            conf = 99
            rt = 99
    
    ## LOG DATA
    exp.addData('prev_resp', prev_resp)    
    exp.addData('resp', resp)
    exp.addData('correct', correct)
    exp.addData('conf', conf)
    exp.addData('rt', rt)
    exp.nextEntry()
    
    return resp, correct, conf, rt


## MISC FUNCTIONS ##

def get_sequence(kappa, nSamples):
    rv = vonmises(kappa, scale=1/(np.pi*0.125))
    r = abs(rv.rvs(size=nSamples))
    x = np.trunc(r+1)
    sample = []
    for ele in x:
        if ele == 1 or ele == 16:
            sample.append(4)
        elif ele == 2 or ele == 15:
            sample.append(3)
        elif ele == 3 or ele == 14:
            sample.append(2)
        elif ele == 4 or ele == 13:
            sample.append(1)
        elif ele == 5 or ele == 12:
            sample.append(-1)
        elif ele == 6 or ele == 11:
            sample.append(-2)
        elif ele == 7 or ele == 10:
            sample.append(-3)
        elif ele == 8 or ele == 9:
            sample.append(-4)
    return sample
