## \file
## \ingroup tutorial_tree
## \notebook
##
## This script displays the TTree data structures.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT
import ctypes
from array import array


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   TPaveLabel,
                   TPaveText,
                   TPavesText,
                   TLine,
                   TArrow,
                   TText,
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
                   kBlue,
                   kRed,
                   kGreen,
)

# globals
from ROOT import (
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)



# void
def tree() :

   #
   global c1
   c1 = TCanvas("c1", "Tree Data Structure", 200, 10, 750, 940) # TCanvas
   c1.Range(0, -0.1, 1, 1.15)
   

   gBenchmark.Start("tree")

   
   #
   branchcolor = 26 # Int_t
   leafcolor = 30 # Int_t
   basketcolor = 42 # Int_t
   offsetcolor = 43 # Int_t


   #
   global title
   title = TPaveLabel(.3, 1.05, .8, 1.13, c1.GetTitle()) # TPaveLabel
   #
   title.SetFillColor(16)
   title.Draw()


   #
   global treePave
   treePave = TPaveText(.01, .75, .15, 1.00) # TPaveText
   #
   treePave.SetFillColor(18)
   treePave.SetTextAlign(12)


   #
   tnt = treePave.AddText("Tree") # TText *
   #
   tnt.SetTextAlign(22)
   tnt.SetTextSize(0.030)


   #
   treePave.AddText("fScanField")
   treePave.AddText("fMaxEventLoop")
   treePave.AddText("fMaxVirtualSize")
   treePave.AddText("fEntries")
   treePave.AddText("fDimension")
   treePave.AddText("fSelectedRows")
   treePave.Draw()


   #
   global farm
   farm = TPavesText(.01, 1.02, .15, 1.1, 9, "tr") # TPavesText


   #
   tfarm = farm.AddText("CHAIN") # TText *
   tfarm.SetTextSize(0.024)

   #
   farm.AddText("Collection")
   farm.AddText("of Trees")
   farm.Draw()


   #
   global llink
   llink = TLine(.15, .92, .80, .92) # TLine
   #
   llink.SetLineWidth(2)
   llink.SetLineColor(1)
   llink.Draw()
   llink.DrawLine(.21, .87, .21, .275)
   llink.DrawLine(.23, .87, .23, .375)
   llink.DrawLine(.25, .87, .25, .805)
   llink.DrawLine(.41, .25, .41, -.025)
   llink.DrawLine(.43, .25, .43, .075)
   llink.DrawLine(.45, .25, .45, .175)


   #
   global branch0
   branch0 = TPaveLabel(.20, .87, .35, .97, "Branch 0") # TPaveLabel
   #
   branch0.SetTextSize(0.35)
   branch0.SetFillColor(branchcolor)
   branch0.Draw()


   #
   global branch1
   branch1 = TPaveLabel(.40, .87, .55, .97, "Branch 1") # TPaveLabel
   #
   branch1.SetTextSize(0.35)
   branch1.SetFillColor(branchcolor)
   branch1.Draw()


   #
   global branch2
   branch2 = TPaveLabel(.60, .87, .75, .97, "Branch 2") # TPaveLabel
   #
   branch2.SetTextSize(0.35)
   branch2.SetFillColor(branchcolor)
   branch2.Draw()


   #
   global branch3
   branch3 = TPaveLabel(.80, .87, .95, .97, "Branch 3") # TPaveLabel
   #
   branch3.SetTextSize(0.35)
   branch3.SetFillColor(branchcolor)
   branch3.Draw()


   #
   global leaf0
   leaf0 = TPaveLabel(.4, .78, .5, .83, "Leaf 0") # TPaveLabel
   #
   leaf0.SetFillColor(leafcolor)
   leaf0.Draw()


   #
   global leaf1
   leaf1 = TPaveLabel(.6, .78, .7, .83, "Leaf 1") # TPaveLabel
   #
   leaf1.SetFillColor(leafcolor)
   leaf1.Draw()


   #
   global leaf2
   leaf2 = TPaveLabel(.8, .78, .9, .83, "Leaf 2") # TPaveLabel
   #
   leaf2.SetFillColor(leafcolor)
   leaf2.Draw()


   #
   global firstevent
   firstevent = TPaveText(.4, .35, .9, .4) # TPaveText
   #
   firstevent.AddText("First event of each basket")
   firstevent.AddText("Array of fMaxBaskets Integers")
   firstevent.SetFillColor(basketcolor)
   firstevent.Draw()


   #
   global basket0
   basket0 = TPaveLabel(.4, .25, .5, .3, "Basket 0") # TPaveLabel
   #
   basket0.SetFillColor(basketcolor)
   basket0.Draw()


   #
   global basket1
   basket1 = TPaveLabel(.6, .25, .7, .3, "Basket 1") # TPaveLabel
   #
   basket1.SetFillColor(basketcolor)
   basket1.Draw()


   #
   global basket2
   basket2 = TPaveLabel(.8, .25, .9, .3, "Basket 2") # TPaveLabel
   #
   basket2.SetFillColor(basketcolor)
   basket2.Draw()
   
   global offset
   offset = TPaveText(.55, .15, .9, .2) # TPaveText
   offset.AddText("Offset of events in fBuffer")
   offset.AddText("Array of fEventOffsetLen Integers")
   offset.AddText("(if variable length structure)")
   offset.SetFillColor(offsetcolor)
   offset.Draw()


   #
   global buffer
   buffer = TPaveText(.55, .05, .9, .1) # TPaveText
   #
   buffer.AddText("Basket buffer")
   buffer.AddText("Array of fBasketSize chars")
   buffer.SetFillColor(offsetcolor)
   buffer.Draw()


   #
   global zipbuffer
   zipbuffer = TPaveText(.55, -.05, .75, .0) # TPaveText
   #
   zipbuffer.AddText("Basket compressed buffer")
   zipbuffer.AddText("(if compression)")
   zipbuffer.SetFillColor(offsetcolor)
   zipbuffer.Draw()


   #
   global ar1
   ar1 = TArrow()
   #
   ar1.SetLineWidth(2)
   ar1.SetLineColor(1)
   ar1.SetFillStyle(1001)
   ar1.SetFillColor(1)
   #
   ar1.DrawArrow(.21, .275, .39, .275, 0.015, "|>")
   ar1.DrawArrow(.23, .375, .39, .375, 0.015, "|>")
   ar1.DrawArrow(.25, .805, .39, .805, 0.015, "|>")
   ar1.DrawArrow(.50, .805, .59, .805, 0.015, "|>")
   ar1.DrawArrow(.70, .805, .79, .805, 0.015, "|>")
   ar1.DrawArrow(.50, .275, .59, .275, 0.015, "|>")
   ar1.DrawArrow(.70, .275, .79, .275, 0.015, "|>")
   ar1.DrawArrow(.45, .175, .54, .175, 0.015, "|>")
   ar1.DrawArrow(.43, .075, .54, .075, 0.015, "|>")
   ar1.DrawArrow(.41, -.025, .54, -.025, 0.015, "|>")


   #
   global ldot
   ldot = TLine(.95, .92, .99, .92) # TLine
   #
   ldot.SetLineStyle(3)
   ldot.Draw()
   ldot.DrawLine(.9, .805, .99, .805)
   ldot.DrawLine(.9, .275, .99, .275)
   ldot.DrawLine(.55, .05, .55, 0)
   ldot.DrawLine(.9, .05, .75, 0)


   #
   global pname
   pname = TText(.46, .21, "fEventOffset") # TText
   #
   pname.SetTextFont(72)
   pname.SetTextSize(0.018)
   #
   pname.Draw()
   #
   pname.DrawText(.44, .11, "fBuffer")
   pname.DrawText(.42, .01, "fZipBuffer")
   pname.DrawText(.26, .84, "fLeaves = TObjArray of TLeaf")
   pname.DrawText(.24, .40, "fBasketEvent")
   pname.DrawText(.22, .31, "fBaskets = TObjArray of TBasket")
   pname.DrawText(.20, 1.0, "fBranches = TObjArray of TBranch")


   #
   global ntleaf
   ntleaf = TPaveText(0.30, .42, .62, .73) # TPaveText
   #
   ntleaf.SetTextSize(0.014)
   ntleaf.SetFillColor(leafcolor)
   ntleaf.SetTextAlign(12)
   #
   ntleaf.AddText("fLen: number of fixed elements")
   ntleaf.AddText("fLenType: number of bytes of data type")
   ntleaf.AddText("fOffset: relative to Leaf0-fAddress")
   ntleaf.AddText("fNbytesIO: number of bytes used for I/O")
   ntleaf.AddText("fIsPointer: True if pointer")
   ntleaf.AddText("fIsRange: True if leaf has a range")
   ntleaf.AddText("fIsUnsigned: True if unsigned")
   ntleaf.AddText("*fLeafCount: points to Leaf counter")
   ntleaf.AddText(" ")
   #
   ntleaf.AddLine(0, 0, 0, 0)
   #
   ntleaf.AddText("fName = Leaf name")
   ntleaf.AddText("fTitle = Leaf type (see Type codes)")
   #
   ntleaf.Draw()


   #
   global types_description
   types_description = TPaveText(.65, .42, .95, .73) # TPaveText
   #
   types_description.SetTextAlign(12)
   types_description.SetFillColor(leafcolor)
   #
   types_description.AddText(" ")
   types_description.AddText("C : a character string")
   types_description.AddText("B : an 8 bit signed integer")
   types_description.AddText("b : an 8 bit unsigned integer")
   types_description.AddText("S : a 16 bit signed short integer")
   types_description.AddText("s : a 16 bit unsigned short integer")
   types_description.AddText("I : a 32 bit signed integer")
   types_description.AddText("i : a 32 bit unsigned integer")
   types_description.AddText("F : a 32 bit floating point")
   types_description.AddText("f : a 24 bit truncated float")
   types_description.AddText("D : a 64 bit floating point")
   types_description.AddText("d : a 24 bit truncated double")
   types_description.AddText("TXXXX : a class name TXXXX")
   #
   types_description.Draw()


   #
   global typecode
   typecode = TPaveLabel(.7, .71, .9, .75, "fType codes") # TPaveLabel
   #
   typecode.SetFillColor(leafcolor)
   typecode.Draw()
   #
   ldot.DrawLine(.4, .78, .30, .73)
   ldot.DrawLine(.5, .78, .62, .73)


   #
   global ntbasket
   ntbasket = TPaveText(0.02, -0.07, 0.35, .25) # TPaveText
   #
   ntbasket.SetFillColor(basketcolor)
   ntbasket.SetTextSize(0.014)
   ntbasket.SetTextAlign(12)
   #
   ntbasket.AddText("fNbytes: Size of compressed Basket")
   ntbasket.AddText("fObjLen: Size of uncompressed Basket")
   ntbasket.AddText("fDatime: Date/Time when written to store")
   ntbasket.AddText("fKeylen: Number of bytes for the key")
   ntbasket.AddText("fCycle : Cycle number")
   ntbasket.AddText("fSeekKey: Pointer to Basket on file")
   ntbasket.AddText("fSeekPdir: Pointer to directory on file")
   ntbasket.AddText("fClassName: 'TBasket'")
   ntbasket.AddText("fName: Branch name")
   ntbasket.AddText("fTitle: TreePave name")
   ntbasket.AddText(" ")
   ntbasket.AddLine(0, 0, 0, 0)
   ntbasket.AddText("fNevBuf: Number of events in Basket")
   ntbasket.AddText("fLast: pointer to last used byte in Basket")
   #
   ntbasket.Draw()
   #
   ldot.DrawLine(.4, .3, 0.02, 0.25)
   ldot.DrawLine(.5, .25, 0.35, -.07)
   ldot.DrawLine(.5, .3, 0.35, 0.25)


   #
   global ntbranch
   ntbranch = TPaveText(0.02, 0.40, 0.18, 0.68) # TPaveText
   #
   ntbranch.SetFillColor(branchcolor)
   ntbranch.SetTextSize(0.015)
   ntbranch.SetTextAlign(12)
   #
   ntbranch.AddText("fBasketSize")
   ntbranch.AddText("fEventOffsetLen")
   ntbranch.AddText("fMaxBaskets")
   ntbranch.AddText("fEntries")
   ntbranch.AddText("fAddress of Leaf0")
   ntbranch.AddText(" ")
   ntbranch.AddLine(0, 0, 0, 0)
   ntbranch.AddText("fName: Branchname")
   ntbranch.AddText("fTitle: leaflist")
   #
   ntbranch.Draw()
   #
   ldot.DrawLine(.2, .97, .02, .68)
   ldot.DrawLine(.35, .97, .18, .68)
   ldot.DrawLine(.35, .87, .18, .40)


   #
   global basketstore
   basketstore = TPavesText(.8, -0.088, 0.952, -0.0035, 7, "tr") # TPavesText
   #
   basketstore.SetFillColor(28)
   basketstore.AddText("Baskets")
   basketstore.AddText("Stores")
   basketstore.Draw()


   gBenchmark.Show("treePave")


   c1.Update()




if __name__ == "__main__":
   tree()
