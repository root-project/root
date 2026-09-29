## \file
## \ingroup tutorial_tree
## \notebook
##
##
## Playing with a Tree which contains variables of type character.
##
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import os
import ctypes
from array import array

import ROOT


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   TPaveText,
                   TArrow,
                   TLegend,
                   THStack,
                   TFile,
                   TCanvas,
                   TH1F,
                   TGraph,
                   TLatex,
)

# maths
from ROOT import (
                   sin,
                   cos,
                   sqrt,
)

# types
from ROOT import (
                   Double_t,
                   Bool_t,
                   Float_t,
                   Int_t,
                   nullptr,
)
#
from ctypes import c_double

# utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args, end="")
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

# constants
from ROOT import (
                   kWhite,
                   kCyan,
                   kYellow,
                   kBlue,
                   kRed,
                   kGreen,
)

# globals
from ROOT import (
                   gDirectory,
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)



# void
def cernstaff() :

   tut_dir = gROOT.GetTutorialDir()  # TString
   Dir = tut_dir.Data() + "/tree/cernstaff.C"
   if not os.path.exists( "cernstaff.root" ) :
      gROOT.SetMacroPath( tut_dir.Data() + "/tree" )
      gROOT.ProcessLine(".x cernbuild.C")
      

   #
   global f, T
   f = TFile("cernstaff.root")  # TFile
   T = f.Get("T")  # (TTree *)


   #
   global c1
   c1 = TCanvas("c1", "CERN staff", 10, 10, 1000, 750); # TCanvas
   c1.Divide(2, 2)


   #
   # Make a table with number of people per Nation & Division.
   #
   c1.cd(1)
   gPad.SetGrid()
   #
   T.Draw("Nation:Division>>hN", "", "text")
   #

   global hN
   hN = gDirectory.Get("hN")  # (TH2F *)
   #
   hN.SetMarkerSize(1.6)
   hN.SetStats(0)
   
   #
   # Make profile of Average cost per Nation.
   #
   c1.cd(2)
   gPad.SetGrid()
   gPad.SetLeftMargin(0.12)
   #
   T.Draw("Cost:Nation>>hNation", "", "prof,goff")
   #

   global hNation
   hNation = gDirectory.Get("hNation")  # (TH1F *)
   #
   hNation.SetTitle("Average Cost per Nation")
   hNation.LabelsOption(">")  # sort by decreasing Bin contents
   hNation.SetMaximum(13000)
   hNation.SetMinimum(7000)
   hNation.SetStats(0)
   hNation.SetMarkerStyle(21)
   #
   hNation.Draw()
   
   #
   # Make stacked plot of Nations versus Grade.
   #
   c1.cd(3)
   gPad.SetGrid()
   #

   global hGrades
   hGrades = THStack("hGrades", "Nations versus Grade"); # THStack
   #
   #
   global hFR
   hFR = TH1F("hFR", "FR", 12, 3, 15)  # TH1F
   hFR.SetFillColor(kCyan)
   hGrades.Add(hFR)
   T.Draw("Grade>>hFR", "Nation==\"FR\"")
   #
   global hCH
   hCH = TH1F("hCH", "CH", 12, 3, 15)  # TH1F
   hCH.SetFillColor(kRed)
   hGrades.Add(hCH)
   T.Draw("Grade>>hCH", "Nation==\"CH\"")
   #
   global hIT
   hIT = TH1F("hIT", "IT", 12, 3, 15)  # TH1F
   hIT.SetFillColor(kGreen)
   hGrades.Add(hIT)
   T.Draw("Grade>>hIT", "Nation==\"IT\"")
   #
   global hDE
   hDE = TH1F("hDE", "DE", 12, 3, 15)  # TH1F
   hDE.SetFillColor(kYellow)
   hGrades.Add(hDE)
   T.Draw("Grade>>hDE", "Nation==\"DE\"")
   #
   global hGB
   hGB = TH1F("hGB", "GB", 12, 3, 15)  # TH1F
   hGB.SetFillColor(kBlue)
   hGrades.Add(hGB)
   T.Draw("Grade>>hGB", "Nation==\"GB\"")
   hGrades.Draw()


   #
   global legend
   legend = TLegend(0.7, 0.65, 0.86, 0.88)  # TLegend
   #
   legend.AddEntry(hGB, "GB", "f")
   legend.AddEntry(hDE, "DE", "f")
   legend.AddEntry(hIT, "IT", "f")
   legend.AddEntry(hCH, "CH", "f")
   legend.AddEntry(hFR, "FR", "f")
   #
   legend.Draw()
   
   # Make a histogram of age distribution.
   c1.cd(4)
   gPad.SetGrid()
   #
   T.Draw("Age")
   T.Draw("Age>>hRetired", "Age>(65-2002+1988)", "same")
   #
   global hRetired
   hRetired = gDirectory.Get("hRetired")  # (TH1F *)
   hRetired.SetFillColor(kRed)
   hRetired.SetFillStyle(3010)
   

   #
   global arrow
   arrow = TArrow(32, 169, 55, 74, 0.03, "|>"); # TArrow
   arrow.SetFillColor(1)
   arrow.SetFillStyle(1001)
   arrow.Draw()

   
   #
   global pt
   pt = TPaveText(0.12, 0.8, 0.55, 0.88, "brNDC"); # TPaveText
   pt.SetFillColor(kWhite)
   pt.AddText("People at CERN in 1988")
   pt.AddText("and retired in 2002")
   pt.Draw()
   
   c1.cd()
   c1.Update()
   c1.Draw()
   


if __name__ == "__main__":
   cernstaff()
