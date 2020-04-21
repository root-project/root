#! /usr/bin/env python

import ROOT
import shutil
import os
import sys

def makeimage(MacroName, ImageName, OutDir, cp, py, batch):
    '''Generates the ImageName output of the macro MacroName'''

    ROOT.gStyle.SetImageScaling(3.)

    if batch:
        ROOT.gROOT.SetBatch(1)

    if py:
        sys.argv = [MacroName]
        exec(open(MacroName).read(), globals())
    else:
        ROOT.gInterpreter.ProcessLine(".x " + MacroName)

    if cp:
        MN = MacroName.split("(")[0]
        MNBase = os.path.basename(MN)
        shutil.copyfile("%s" %MN,"%s/macros/%s" %(OutDir,MNBase))

    ImageNum = 0
    s = open ("ImagesSizes.dat","w")

    canvases = ROOT.gROOT.GetListOfCanvases()
    for ImageNum,can in enumerate(canvases):
        ImageNum += 1
        can.SaveAs("%s/html/pict%d_%s" %(OutDir,ImageNum,ImageName))
        cw = can.GetWindowWidth()
        s.write("%d\n" %cw)

    s.close()

    f = open ("NumberOfImages.dat","w")
    f.write("%d\n" %ImageNum)
    f.close()

if __name__ == "__main__":
    from sys import argv
    makeimage(argv[1], argv[2], argv[3], bool(argv[4]), bool(argv[5]), bool(argv[6]))
