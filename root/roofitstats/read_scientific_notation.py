import os
import sys
sys.argv.append( '-b-' )
import ROOT as r

def CreateFile():
    x = r.RooRealVar('x','x',1000000,10000,1000000)
    aset= r.RooArgSet(x)
    aset.writeToFile('test.txt')
    return

def LoadFile():
    x = r.RooRealVar('x','x',10,1,100)
    ws = r.RooWorkspace('ws')
    ws.__getattribute__('import')(x)
    print 'x before reading'
    x.Print()
    ws.allVars().readFromFile('test.txt')
    print 'x after reading'
    ws.var('x').Print()
    return

if not os.path.isfile('test.txt'): CreateFile()
LoadFile()
