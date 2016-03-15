#! /usr/bin/env python

import ROOT
import os

def makeimage(MacroName, ImageName, OutDir, cp, py):
    '''Generates the ImageName output of the macro MacroName'''
    if py: exec(MacroName)
    else: ROOT.gInterpreter.ProcessLine(".x " + MacroName)

    if cp:
        MN = MacroName.split("(")[0]
        os.cp("%s","%s/macros", MN, OutDir)

    canvases = ROOT.gROOT.GetListOfCanvases()
    for ImageNum,can in enumerate(canvases):
        can.SaveAs("%s/html/pict%d_%s" %(OutDir,ImageNum,ImageName)

    with open("NumberOfImages.dat",'w') as f:
        f.Write("%d\n" %ImageNum)

if __name__ == "__main__":
    from sys import argv
    makeimage(argv[1], argv[2], argv[3], bool(argv[4]), bool(argv[5]))

