## \file
## \ingroup tutorial_graphs
## \notebook -js
##
## This script illustrates how to use the time axis on a TGraph-object
## with: data read from a text file containing the SWAN usage
## statistics during July 2017.
## We exploit TDataFrame-class for reading data from a file.
##
## \macro_image
## \macro_code
##
## \authors Danilo Piparo, Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TDatime = ROOT.TDatime

TH1F = ROOT.TH1F
TGraph = ROOT.TGraph
TLatex = ROOT.TLatex

TCanvas = ROOT.TCanvas
TPaveLabel = ROOT.TPaveLabel
TPavesText = ROOT.TPavesText
TPaveText = ROOT.TPaveText
TText = ROOT.TText
TArrow = ROOT.TArrow
TWbox = ROOT.TWbox
TPad = ROOT.TPad
TBox = ROOT.TBox
TPad = ROOT.TPad
#maths
sin = ROOT.sin
cos = ROOT.cos
sqrt = ROOT.sqrt

#types
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
nullptr = ROOT.nullptr
c_double = ctypes.c_double

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args )
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer
def Form(string, *args):
   return string % args

#constants
kBlue = ROOT.kBlue
kRed = ROOT.kRed
kGreen = ROOT.kGreen

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT



# void
def timeSeriesFromCSV_TDF() :
   # Open csv data file SWAN2017.dat.
   # This csv contains the usage statistics of one CERN IT
   # service: SWAN, during two weeks. We would like to plot this data with
   # ROOT so we can get some conclusions from it.
   Dir = gROOT.GetTutorialDir()
   Dir.Append("/graphs/")
   Dir.ReplaceAll("/./", "/")
   
   # Read the data from the file using TDataFrame. We do not have headers and
   # we would like the delimiter to be a space.
   global tdf
   tdf = ROOT.RDF.FromCSV(Form("%sSWAN2017.dat", Dir.Data()), False, ' ')
   
   # We now prepare the graph input.
   global d, timeStamps, values
   d = tdf.Define("TimeStamp", "auto s = string(Col0) + ' ' +  Col1; return (float) TDatime(s.c_str()).Convert();").Define("Value", "(float)Col2")
   timeStamps = d.Take["Float_t"]("TimeStamp")
   values = d.Take["Float_t"]("Value")
   
   # Create the time graph.

   global g
   g = TGraph(values.size(), timeStamps.data(), values.data())
   g.SetTitle("SWAN Users during July 2017;Time;Number of Sessions")
   
   # Draw the graph.

   global c
   c = TCanvas("c", "c", 950, 500)
   c.SetLeftMargin(0.07)
   c.SetRightMargin(0.04)
   c.SetGrid()
   g.SetLineWidth(3)
   g.SetLineColor(kBlue)
   g.Draw("al")
   g.GetYaxis().CenterTitle()
   
   # Make the X axis labelled with time.
   xaxis = g.GetXaxis()
   xaxis.SetTimeDisplay(1)
   xaxis.CenterTitle()
   xaxis.SetTimeFormat("%a %d")
   xaxis.SetTimeOffset(0)
   xaxis.SetNdivisions(-219)
   xaxis.SetLimits(TDatime(2017, 7, 3, 0, 0, 0).Convert(), TDatime(2017, 7, 22, 0, 0, 0).Convert())
   xaxis.SetLabelSize(0.025)
   xaxis.CenterLabels()
   


if __name__ == "__main__":
   timeSeriesFromCSV_TDF()
