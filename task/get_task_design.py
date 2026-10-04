import numpy as np
import pandas as pd
import random
import math
from scipy.stats import truncexpon, vonmises
from helper_functions import get_sequence

class TaskDesign:
    def __init__(self, participant_number, practice=False):
        if practice:
            self.nBlocks = 8
            self.nTrials = 54
        else:
            self.nBlocks = 8
            self.nTrials = 96
        self.participant_number = participant_number
        self.cb_group = self.counterbalance()
        self.taskid = self.get_taskid(practice)
        self.epimap = self.get_epimap(practice)
        self.condid = self.get_condid(practice)
        self.epinum, self.epipos = self.get_epinum(self.nBlocks, self.nTrials)
        self.seqdir = self.get_seqdir()
        self.seqlvl, self.seqllr = self.get_evidence(practice)
        self.correct_HC, self.correct_LC, self.incorrect_HC, self.incorrect_LC = self.get_correct_resp()
        self.stimList = self.get_stimList()
    
    def counterbalance(self):
        ptp = self.participant_number
        if ptp in range(1,99999,4):
            return 1
        elif ptp in range(2,99999,4):
            return 2
        elif ptp in range(3,99999,4):
            return 3
        else:
            return 4

    def get_taskid(self, practice):
        # taskid: which task to perform
        # 0 = observer, 1 = actor
        if (self.cb_group % 2) == 0:
            if practice:
                return [0,0,0,0,1,1,1,1]
            return [0,0,1,1,0,0,1,1]
        else:
            if practice:
                return [1,1,1,1,0,0,0,0]
            return [1,1,0,0,1,1,0,0]
        
    def get_epimap(self, practice):
        # epimap: color-key mappings
        # 0 = BlueOrange/drawBlue, 1 = OrangeBlue/drawOrange
        if self.cb_group == 1:
            if practice:
                return [0,1,0,1,0,1,0,1]
            return [0,1,0,1,0,1,0,1]
        elif self.cb_group == 2:
            if practice:
                return [1,0,1,0,1,0,1,0]
            return [1,0,1,0,1,0,1,0]
        elif self.cb_group == 3:
            if practice:
                return [0,1,0,1,0,1,0,1]
            return [0,1,1,0,0,1,1,0]
        elif self.cb_group == 4:
            if practice:
                return [1,0,1,0,1,0,1,0]
            return [1,0,0,1,1,0,0,1]
        
    def get_condid(self, practice):
        # condid: high or low volatility
        # -1 = practice, 0 = low, 1 = high
        if practice:
            return [-1,-1,-1,-1,-1,-1,-1,-1]
        else:
            if self.cb_group == 1:
                return [0,1,1,0,1,0,0,1]
            elif self.cb_group == 2:
                return [1,0,0,1,0,1,1,0]
            elif self.cb_group == 3:
                return [0,1,0,1,1,0,1,0]
            elif self.cb_group == 4:
                return [1,0,1,0,0,1,0,1]
        
    def get_epinum(self, nBlocks, nTrials):
        # epinum: episode number within block
        # epipos: trial number within episode
        epinum = []
        epipos = []
        
        epinum_block_list, epipos_block_list = [], []
        for iBl in range(nBlocks//2):
            # determine hazard rate based on condition id
            if self.condid[iBl] == -1:
                hazard = 1/9
            elif self.condid[iBl] == 0:
                hazard = 1/12
            elif self.condid[iBl] == 1:
                hazard = 1/6
            nepi = int(hazard*nTrials)
            davg = int(nTrials/nepi)
            upper, lower = 24, 4
            x = truncexpon(b=upper, loc=lower, scale=davg-lower)
            # create block episodes
            epinum_block = []
            epipos_block = []
            num_counter = 0
            eps = 0
            while not eps == nTrials:
                g = x.rvs(int(nepi))
                if any(guess > 23 for guess in g):
                    continue
                eps = sum([round(guess) for guess in g])
            for ep in g:
                pos_counter = 0
                num_counter += 1
                for _ in range(round(ep)):
                    pos_counter += 1
                    epipos_block.append(pos_counter)
                    epinum_block.append(num_counter)
            epinum_block_list.append(epinum_block)
            epipos_block_list.append(epipos_block)
            
        # put in right order
        order = [0,1,2,3,1,0,3,2]
        for i in order:
            epinum.append(epinum_block_list[i])
            epipos.append(epipos_block_list[i])
            
        return epinum, epipos
    
    def get_seqdir(self):
        # seqdir: hidden state of trial
        # 0 = blue/blueLeft, 1 = orange/orangeLeft
        seqdir = []
        for iBl in range(self.nBlocks):
            seqdir_block = []
            current_dir = random.sample([0,1], k=1)[0]
            seqdir_block.append(current_dir)
            for iTr in range(1,self.nTrials):
                if self.epinum[iBl][iTr-1]==self.epinum[iBl][iTr]:
                    seqdir_block.append(current_dir)
                else:
                    current_dir = int(current_dir==0)
                    seqdir_block.append(current_dir)
            seqdir.append(seqdir_block)
            
        return seqdir
        
    def get_evidence(self, practice):
        # seqlvl: evidence per sample (8x96x4 list)
        # seqllr: evidence per trial (8x96 list)
        # range: -1 (seqdir=0) to 1 (seqdir=1) in steps of .25
        seqlvl = []
        seqllr = []
        recode_dict = {4: 0.9749, 3: 0.9011, 2: 0.6899, 1: 0.3735, -1: -0.3735, -2: -0.6899, -3: -0.9011, -4: -0.9749}
        nSamples = 4
        if practice:
            kappa = 1.0
        else:
            kappa = 0.5
            
        # make sample list and shuffled sample list
        seqlvl_block_list, seqllr_block_list = [], []
        for iBl in range(self.nBlocks//2):
            if iBl>1:
                kappa = 0.5
            seqlvl_block = []
            seqllr_block = []
            for iTr in range(self.nTrials):
                sample = get_sequence(kappa, nSamples)
                seqlvl_block.append(sample)
                seqllr_block.append(math.fsum([recode_dict[s] for s in sample]))
            seqlvl_block_list.append(seqlvl_block)
            seqllr_block_list.append(seqllr_block)
        
        # put in right order
        if practice:
            order = [0,1,2,3,0,1,2,3]
        elif self.cb_group<3:
            order = [0,1,1,0,3,2,2,3]
        elif self.cb_group>2:
            order = [0,1,0,1,2,3,2,3]
        for i in order:
            seqlvl.append(seqlvl_block_list[i])
            seqllr.append(seqllr_block_list[i])
        
        return seqlvl, seqllr
        
    def get_correct_resp(self):
        # which response is correct (left/right)
        correct_HC = []
        correct_LC = []
        incorrect_HC = []
        incorrect_LC = []
        for iBl in range(self.nBlocks):
            correct_HC_block = []
            correct_LC_block = []
            incorrect_HC_block = []
            incorrect_LC_block = []
            for iTr in range(self.nTrials):
                if self.seqdir[iBl][iTr] == self.epimap[iBl]:
                    correct_HC_block.append('e')
                    correct_LC_block.append('r')
                    incorrect_HC_block.append('p')
                    incorrect_LC_block.append('o')
                else:
                    correct_HC_block.append('p')
                    correct_LC_block.append('o')
                    incorrect_HC_block.append('e')
                    incorrect_LC_block.append('r')
            correct_HC.append(correct_HC_block)
            correct_LC.append(correct_LC_block)
            incorrect_HC.append(incorrect_HC_block)
            incorrect_LC.append(incorrect_LC_block)
            
        return correct_HC, correct_LC, incorrect_HC, incorrect_LC
    
    def get_stimList(self):
        stimList = []
        for iBl in range(self.nBlocks):
            block_stimList = []
            for iTr in range(self.nTrials):
                block_stimList.append({'participant_number': self.participant_number,\
                'blocknum': iBl+1, 'trialnum': iTr+1, 'nTrials': self.nTrials,\
                'taskid': self.taskid[iBl], 'epimap': self.epimap[iBl], 'condid': self.condid[iBl],\
                'epinum': self.epinum[iBl][iTr], 'epipos': self.epipos[iBl][iTr],\
                'seqdir': self.seqdir[iBl][iTr], 'seqlvl': self.seqlvl[iBl][iTr], 'seqllr': self.seqllr[iBl][iTr],\
                'correct_HC': self.correct_HC[iBl][iTr], 'correct_LC': self.correct_LC[iBl][iTr],\
                'incorrect_HC': self.incorrect_HC[iBl][iTr], 'incorrect_LC': self.incorrect_LC[iBl][iTr]})
            stimList.append(block_stimList)
        return stimList