## \file
## \ingroup tutorial_graphs
## \notebook -js
##
## This script illustrates how to use the time axis on a TGraph-object
## with data read from a text file containing the SWAN(Service for Web-based Analysis)
## usage statistics during July 2017.
##
## \macro_image
## \macro_code
##
## \authors Danilo Piparo, Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#standard libary
std = ROOT.std
fopen = std.fopen
fgets = std.fgets
sscanf = std.sscanf
strncpy = std.strncpy
fclose = std.fclose

#classes
TDatime = ROOT.TDatime

TCanvas = ROOT.TCanvas
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
c_float = ctypes.c_float

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
create_string_buffer = ctypes.create_string_buffer


#constants
kBlue = ROOT.kBlue
kRed = ROOT.kRed
kGreen = ROOT.kGreen

#globals
gInterpreter = ROOT.gInterpreter
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT

#c-integration
ProcessLine = gInterpreter.ProcessLine

#system utils
Remove = gROOT.Remove

#utils from std
ProcessLine(""" 
auto fopen_Py( const char* name, const char* mode ){
   return std::fopen(name, mode);
} 
""")
ProcessLine("""
auto fgets_Py( char * s, int n, _IO_FILE* stream ){
   return std::fgets( s, n , stream );
}
""")
ProcessLine("""
auto sscanf_Py( const char * s, const char * format, Float_t & destination ){
   return std::sscanf( s, format, destination );
}
""")
ProcessLine("""
auto strncpy_Py( char * dest, char * restrict, unsigned long n){
   return std::strncpy( dest, restrict, n);
   }
""")
fopen_Py = ROOT.fopen_Py
fgets_Py = ROOT.fgets_Py
sscanf_Py = ROOT.sscanf_Py
strncpy_Py = ROOT.strncpy_Py


# void
def timeSeriesFromCSV() :
   # Open the csv data file. This csv-file contains the usage statistics of one CERN-IT
   # service, SWAN, during two weeks. We would like to plot this data with
   # ROOT to draw get some conclusions from it.
   global Dir
   Dir = gROOT.GetTutorialDir()
   print("1th get", Dir)
   Dir.Append("/graphs/")
   Dir.ReplaceAll("/./", "/")
   print(Dir)


   global f
   f = fopen_Py(Form("%sSWAN2017.dat", Dir.Data()), "r")
   # If we want to run this script again
   # and since Dir is a pointer to gROOT.GetTutorialDir(),
   # we have to retreive its original path.
   Dir.ReplaceAll("/graphs/", "") 
   
   # Create the time graph

   global g
   g = TGraph()
   g.SetTitle("SWAN Users during July 2017;Time;Number of Sessions")
   
   # Read the data and fill the graph with time along the X axis and number
   # of users along the Y axis
   #line = " "*80 # char

   #global line, v, dt
   line = create_string_buffer(80)
   v = c_float()
   #dt = " "*20 # char
   dt = create_string_buffer(20)
   i = 0
   while (fgets_Py(line, 80, f)) :
      sscanf_Py(line[20:], "%f", v)
      strncpy_Py(dt, line, 18)
      #Debug: print("dt ", dt.value) 
      #dt[:19] = c_char(0) #'\0' # Unnecessary. Derived from C-version.
      print("v", v.value)
      g.SetPoint(i, TDatime(dt.value).Convert(), v.value)
      i += 1
      
   fclose(f)
   
   # Draw the graph

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
   global xaxis
   xaxis = g.GetXaxis() # TAxis
   xaxis.SetTimeDisplay(1)
   xaxis.CenterTitle()
   xaxis.SetTimeFormat("%a %d")
   xaxis.SetTimeOffset(0)
   xaxis.SetNdivisions(-219)
   xaxis.SetLimits(TDatime(2017, 7, 3, 0, 0, 0).Convert(), 
                   TDatime(2017, 7, 22, 0, 0, 0).Convert()
                   )
   xaxis.SetLabelSize(0.025)
   xaxis.CenterLabels()
   


if __name__ == "__main__":
   timeSeriesFromCSV()
