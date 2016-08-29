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

import sys, string, os
from ROOT import TFile, TNtuple


ifn = os.path.expandvars("${ROOTSYS}/tutorials/pyroot/aptuple.txt")
ofn = 'aptuple.root'

print 'opening file', ifn, '...'
infile = open( ifn, 'r' )
lines  = infile.readlines()
title  = lines[0]
labels = string.split( lines[1] )

print 'writing file', ofn, '...'
outfile = TFile( ofn, 'RECREATE', 'ROOT file with an NTuple' )
ntuple  = TNtuple( 'ntuple', title, string.join( labels, ':') )

for line in lines[2:]:
    words = string.split( line )
    row = map( float, words )
    apply( ntuple.Fill, row )

outfile.Write()

print 'done'
