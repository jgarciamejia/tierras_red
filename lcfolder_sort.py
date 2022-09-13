#!/usr/bin/env python

from __future__ import print_function

import logging
import argparse 
import numpy 
import sys
import os

date = sys.argv[1]
target = sys.argv[2]
folder1 = sys.argv[3]
folder2 = sys.argv[4]

basepath = '/data/tierras/lightcurves/'
targetpath = os.path.join(basepath,date,target)
folder1path = os.path.join(targetpath,folder1)
folder2path = os.path.join(targetpath,folder2)

if not os.path.exists(targetpath):
    print ('Creating {} Directory'.format(targetpath))
    os.system('mkdir {}'.format(targetpath))
    os.system('mv {} {}'.format(os.path.join(basepath,date,"*"),targetpath))
    os.system('mkdir {}'.format(folder1path))
    os.system('mv {} {}'.format(os.path.join(targetpath,"*"),folder1path))
    os.system('mkdir {}'.format(folder2path))

else:
    if not os.path.exists(folder1path):   
        os.system('mkdir {}'.format(folder1path))
        os.system('mv {} {}'.format(os.path.join(targetpath,"*"),folder1path))
        os.system('mkdir {}'.format(folder2path))
