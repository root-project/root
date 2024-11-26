## \file
## \ingroup tutorial_pyroot
## \notebook -nodraw
## Build ROOT Ntuple from other source.
## This program reads the `aptuple.txt' file row by row, then creates
## the Ntuple by adding row by row.
##
## \macro_output
## \macro_code
##
## \author Wim Lavrijsen

import sys, os
from ROOT import TFile, TNtuple, TROOT


ifn = os.path.join(str(TROOT.GetTutorialDir()), 'pyroot', 'aptuple.txt')
ofn = 'aptuple.root'

print('opening file %s ...' % ifn)
infile = open( ifn, 'r' )
lines  = infile.readlines()
title  = lines[0]
labels = lines[1].split()

print('writing file %s ...' % ofn)
outfile = TFile( ofn, 'RECREATE', 'ROOT file with an NTuple' )
ntuple  = TNtuple( 'ntuple', title, ':'.join( labels ) )

for line in lines[2:]:
    words = line.split()
    row = map( float, words )
    ntuple.Fill(*row)

outfile.Write()

print('done')
