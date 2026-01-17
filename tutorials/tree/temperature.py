## \file
## \ingroup tutorial_tree
##
## This tutorial illustrates how to use the highlight mode with trees.
## It first creates a TTree from a temperature data set in Prague between 1775
## and 2004. Then it defines three pads representing the temperature per year,
## month and day. Thanks to the highlight mechanism it is possible to explore the
## data set only by moving the mouse on the plots: 
##
##    - Movements on the years' plot will update
##      the months' and days' plot. 
##    - Movements on the months plot will update
##      the days plot. 
##    - Movements on the days' plot will display
##      the exact temperature
##      for a given day.
##
## Enjoy!
##
## \macro_code
##
## \date March 2018
## \author Jan Musinsky
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
                   Form, # 
                   TTree,
                   TVirtualPad,
                   TProfile,
                   TCanvas,
                   TH1F,
                   TPyDispatcher,
                   TGraph,
                   TLatex,
                   TString,
                   TObject,
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

def TString_Format(ftm, *args):
   return str( ftm % args )
TString.Format = TString_Format


# constants
from ROOT import (
                   kGray,
                   kBlue,
                   kRed,
                   kGreen,
                   # 
                   # - -  
                   kFullDotMedium,
)


# globals
from ROOT import (
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)





# global variables
#
year       = Int_t()       #              
month      = Int_t()       #              
day        = Int_t()       #              
customhb   = Int_t(-2)     #              


# global ROOT objects
#
tree       = nullptr # TTree()       # *            
hYear      = nullptr # TProfile()    # *            
hMonth     = nullptr # TProfile()    # *            
hDay       = nullptr # TProfile()    # *            
Canvas     = nullptr # TCanvas()     # *            
info       = nullptr # TLatex()      # *            


# parameters: Ranges for year, month, day and temperature.
#
rYear  = [ 0, 0, 0 ]            # from tree/data
rMonth = [12, 1, 13 ]           # Int_t [3]
rDay   = [31, 1, 32 ]           # Int_t [3]
rTemp  = [55.0, -20.0, 35.0 ]   # Double_t [3]




# void
def HighlightDay(xhb : Int_t) :

   global year, month, day, customhb
   global tree, hYear, hMonth, hDay, Canvas, info
   global rYear, rMonth, rDay, rTemp

   if ( not info ) :
      #
      info =  TLatex()  # new
      info.SetTextSizePixels(25)
      #
      Canvas.cd(3)
      info.Draw()
      Canvas.Update()
      #
      gPad.Update()
      
   
   if (xhb != customhb)  :
      day = xhb
      

   temp = TString.Format(" %5.1f #circC", hDay.GetBinContent(day))  # TString
   if (hDay.GetBinEntries(day) == 0)  :
      temp = " " # TString
      
   m = " "  # TString
   if (month > 0)  :
      m = TString.Format("-%02d", month)
      
   d = " "  # TString
   if (day > 0)  :
      d = TString.Format("-%02d", day)
      
   info.SetText(2.0,
                hDay.GetMinimum() * 0.8,
                TString.Format(
                   "%4d%s%s%s",
                   year,
                   m,
                   d,
                   temp,
                   #m.Data(),
                   #d.Data(),
                   #temp.Data(),
                   ),
                )

   Canvas.GetPad(3).Modified()
   

# void
def HighlightMonth(xhb : Int_t) :

   global year, month, day, customhb
   global tree, hYear, hMonth, hDay, Canvas, info
   global rYear, rMonth, rDay, rTemp

   if ( not hDay ) :
      #
      hDay =  TProfile("hDay",
                       "; day; temp, #circC",
                       rDay[0],
                       rDay[1],
                       rDay[2],
                       )  # new
      #
      hDay.SetMinimum(rTemp[1])
      hDay.SetMaximum(rTemp[2])
      hDay.GetYaxis().SetNdivisions(410)
      hDay.SetFillColor(kGray)
      hDay.SetMarkerStyle(kFullDotMedium)
      #
      Canvas.cd(3)
      hDay.Draw("HIST, CP")
      Canvas.Update()
      #
      gPad.Update()
      hDay.SetHighlight()
      
   
   if (xhb != customhb)  :
      month = xhb
      
   #
   tree.Draw("T:DAY>>hDay",
             TString.Format("MONTH==%d && YEAR==%d",
                month,
                year),
             "goff",
             )

   #
   hDay.SetTitle(
                 TString.Format("temperature by day (month = %02d, year = %d)",
                    month,
                    year,
                    ),
                 )

   #
   Canvas.GetPad(3).Modified()
   
   #
   # Custom call for HighlightDay function.
   HighlightDay(customhb)  
   

# void
def HighlightYear(xhb : Int_t) :

   global year, month, day, customhb
   global tree, hYear, hMonth, hDay, Canvas, info
   global rYear, rMonth, rDay, rTemp
   
   if ( not hMonth ) :
      #
      hMonth = TProfile("hMonth",
                        "; month; temp, #circC",
                        rMonth[0],
                        rMonth[1],
                        rMonth[2],
                        )  # new
      #
      hMonth.SetMinimum(rTemp[1])
      hMonth.SetMaximum(rTemp[2])
      hMonth.GetXaxis().SetNdivisions(112)
      hMonth.GetXaxis().CenterLabels()
      hMonth.GetYaxis().SetNdivisions(410)
      hMonth.SetFillColor(kGray + 1)
      hMonth.SetMarkerStyle(kFullDotMedium)
      #
      Canvas.cd(2).SetGridx()
      hMonth.Draw("HIST, CP")
      Canvas.Update()

      gPad.Update()


      #
      # Connect highllight.
      #
      hMonth.SetHighlight()
      
   
   #
   year = xhb - 1 + rYear[1]
   tree.Draw("T:MONTH>>hMonth", "YEAR==%d"%year, "goff")
   hMonth.SetTitle( TString.Format( "temperature by month (year = %d)", year ))
   #
   Canvas.GetPad(2).Modified()
   
   #
   # Custom call HighlightMonth function.
   HighlightMonth(customhb)  

   #
   #
   Canvas.GetPad(2).Modified()
   #
   

# void
def HighlightTemp(pad : TVirtualPad, obj : TObject, xhb : Int_t ) :

   global year, month, day, customhb
   global tree, hYear, hMonth, hDay, Canvas, info
   global rYear, rMonth, rDay, rTemp

   #
   if (obj == hYear)  :
      HighlightYear(xhb)
      
   if (obj == hMonth)  :
      HighlightMonth(xhb)
      
   if (obj == hDay)  :
      HighlightDay(xhb)
      
   #
   Canvas.Update()
   

# void
def temperature() :

   global year, month, day, customhb
   global tree, hYear, hMonth, hDay, Canvas, info
   global rYear, rMonth, rDay, rTemp

   #
   # Read file (data from Global Historical Climatology Network).
   # Note:
   #       Data format: YEAR/I:MONTH/I:DAY/I:T/F
   #
   tree     =  TTree("tree", "GHCN-Daily")  # new
   #
   
   #
   # Read file $ROOTSYS/tutorials/tree/temperature_Prague.dat
   #
   tutDir = gROOT.GetTutorialDir()  # auto
   Dir = TString( tutDir ) 
   Dir.Append("/tree/")
   Dir.ReplaceAll("/./", "/")
   #
   if (tree.ReadFile("%stemperature_Prague.dat" %  Dir.Data() ) == 0)  :
      raise FileNotFoundError( "stemperature_Prague.dat",
                               "Not found at",
                               Dir.Data(),
                             ) 
      return
      
   
   #
   # Compute range of years.
   #
   # First year.
   tree.GetEntry( 0 )
   rYear[1] = Int_t( tree.GetLeaf("YEAR").GetValue() )  
   #
   # Last year.
   tree.GetEntry( tree.GetEntries() - 1 )
   rYear[2] = Int_t( tree.GetLeaf("YEAR").GetValue() ) 
   rYear[2] = rYear[2] + 1
   #
   # Its Difference.
   rYear[0] = rYear[2] - rYear[1]

   
   #
   # Create a TProfile for the average temperature by years.
   #
   hYear    =  TProfile("hYear",

                        "temperature (average) by year; year; temp, #circC",
                        rYear[0],
                        rYear[1],
                        rYear[2],

                        )  # new

   #
   tree.Draw("T:YEAR>>hYear", "", "goff")
   #
   hYear.SetMaximum(hYear.GetMean(2) * 1.50)
   hYear.SetMinimum(hYear.GetMean(2) * 0.50)
   hYear.GetXaxis().SetNdivisions(410)
   hYear.GetYaxis().SetNdivisions(309)
   hYear.SetLineColor(kGray + 2)
   hYear.SetMarkerStyle(8)
   hYear.SetMarkerSize(0.75)
   

   #
   # Draw the average temperature by years with highlight. 
   #
   gStyle.SetOptStat("em")
   #
   Canvas =  TCanvas("Canvas", "Canvas", 0, 0, 700, 900)  # new
   #
   # Not to use:
   # Canvas.HighlightConnect("HighlightTemp(TVirtualPad*,TObject*,Int_t,Int_t)")
   # Instead:
   PyD_HighlightTemp = TPyDispatcher( HighlightTemp )
   Canvas.Connect("Highlighted(TVirtualPad*,TObject*,Int_t,Int_t))",
                  "TPyDispatcher",
                  PyD_HighlightTemp,
                  "Dispatch( TVirtualPad*, TObject*, Int_t )", 
                  )
   #
   Canvas.Divide(1, 3, 0.001, 0.001)
   Canvas.cd(1)
   #
   hYear.Draw("HIST, LP")
   Canvas.Update()
   gPad.Update()
   # error: gPad is not loading unless Canvas is updated. This doen't happen in .C .
   # gPad.Update()
   #
   # Note:
   #      Take care in the slot argument; we only use one argument for Int_t.
   #      ROOT doesn't support yet a `.Dispatch` method for:
   #      "TPyDispatcher::Dispatch( TVirtualPad*, TObject*, Int_t, Int_t)" 
   #      Thus, we should omit the "yhb" argument in the "Highlighted" function,
   #      which is present in the .C version.
   #
   

   #
   # Connect the highlight procedure to the temperature profile.
   #
   hYear.SetHighlight()
   


if __name__ == "__main__":
   temperature()
