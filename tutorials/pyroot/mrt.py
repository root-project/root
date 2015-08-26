"""
   Build ROOT Ntuple from other source.   
   This program reads the `aptuple.txt' file row by row, then creates
   the Ntuple by adding row by row.
"""

import sys, string
from ROOT import TFile, TNtuple


ifn = 'aptuple.txt'
ofn = 'aptuple.root'

print 'opening file', ifn, '...'
infile = open( 'aptuple.txt', 'r' )
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
