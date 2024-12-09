## \file
## \ingroup tutorial_graphs
## \notebook
##
## This example compares: (a)what the system time function gmtime and localtime give,
## (b)with what TGaxis-class gives. 
## It can be used as test reference to check if TGaxis-class is working
## properly well.
##
## The original code was developed by Philippe Gras (from CEA Saclay - IRFU/SEDI).
## Enjoy!
##
## \macro_image
## \macro_code
##
## \authors Philippe Gras, Bertrand Bellenot, Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#standard library
std = ROOT.std
time_t = std.time_t
tm = std.tm
localtime = std.localtime
gmtime = std.gmtime
strftime = std.strftime

TAxis = ROOT.TAxis 
TGaxis = ROOT.TGaxis 
TCanvas = ROOT.TCanvas 
TString = ROOT.TString 
TLine = ROOT.TLine 
TLatex = ROOT.TLatex 
ctime = ROOT.ctime 
#cstdio = ROOT.cstdio 

#classes
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
UInt_t = ROOT.UInt_t
Bool_t = ROOT.Bool_t
nullptr = ROOT.nullptr
c_double = ctypes.c_double
c_long = ctypes.c_long

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args )
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer
create_string_buffer = ctypes.create_string_buffer
sizeof = len

#constants
kOrange = ROOT.kOrange
kBlue = ROOT.kBlue
kRed = ROOT.kRed
kGreen = ROOT.kGreen

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT

#c-integration
ProcessLine = gROOT.ProcessLine

#std redefitions for python
ProcessLine("""
auto strftime_Py(char* s, unsigned long maxsize, char* format, const tm* tp){
   return std::strftime( s, maxsize, format, tp );
   }
""")
strftime_Py = ROOT.strftime_Py

# TString
def stime(t : UInt_t, utc : Bool_t = False, display_time_zone : Bool_t = True) :
#def stime(t : c_long, utc : Bool_t = False, display_time_zone : Bool_t = True) :
#def stime(t : UInt_t, utc : Bool_t = False, display_time_zone : Bool_t = True) :
#def stime(t : time_t, utc : Bool_t = False, display_time_zone : Bool_t = True) :

   tt = tm()
   t = c_long( t )
   if (utc) : tt = gmtime(t)
   else     : tt = localtime(t)

   #buf = " "*256
   buf = create_string_buffer(256) #
   if (display_time_zone) : strftime_Py(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S %Z", tt)
   else:                    strftime_Py(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tt)

   return TString(buf)
   


# TCanvas
def timeonaxis3() :

   global c   
   c = TCanvas("c", "c")

   #scale paramter 
   f = 1.8
   

   global tex1
   tex1 = TLatex()
   tex1.SetNDC()
   tex1.SetTextFont(102)
   tex1.SetTextSize(0.055*f)
   

   global tex3
   tex3 = TLatex()
   tex3.SetNDC()
   tex3.SetTextFont(102)
   tex3.SetTextSize(0.07*f)
   tex3.SetTextColor(kBlue+2)
   

   global tex2
   tex2 = TLatex()
   tex2.SetNDC()
   tex2.SetTextFont(102)
   tex2.SetTextSize(0.07*f)
   tex2.SetTextColor(kOrange+3)
   
   offset = [ 0, 0, 1325376000, 1341100800 ]
   t = [ 1331150400, 1336417200, 0, 36000 ]
   
   c.SetTopMargin(0)
   c.SetBottomMargin(0)
   c.SetLeftMargin(0)
   c.SetRightMargin(0)
   c.Divide(2, 4, -1, -1)

   global l
   l = TLine()
   l.DrawLine(0.5, 0, 0.5, 1.)
   
   global p, p_list
   global ga, ga_list
   global a, a_list
   p_list = []
   ga_list = []
   a_list = []
   #   for (int i = 0; i < 4; ++i) {
   #      for (int gmt = 0; gmt < 2; ++gmt) {
   for i in range(0, 4, 1):
      for gmt in range(0, 2, 1):

         #opt = (gmt ? "gmt" : "local")
         opt = "gmt" if gmt else "local"

         p = c.cd(2*i + gmt + 1)
         p.SetTopMargin(0)
         p.SetBottomMargin(0)
         p.SetLeftMargin(0)
         p.SetRightMargin(0)
         p.SetFillStyle(4000)
         p_list.append( p )
         
         ga = TGaxis (.4, .25, 5., .25, t[i], t[i] + 1, 1, "t")
         ga.SetTimeFormat("TGaxis label: #color[2]{%Y-%m-%d %H:%M:%S}")
         ga.SetLabelFont(102)
         ga.SetLabelColor(kBlue+2)
         
         ga.SetTimeOffset(offset[i], opt)
         ga.SetLabelOffset(0.04*f)
         ga.SetLabelSize(0.07*f)
         ga.SetLineColor(0)
         ga.Draw()
         ga_list.append( ga )
         
         # Get offset string of axis time format: there is not accessor
         # to time format in TGaxis.
         # Assumes TAxis use the same format.
         a = TAxis(10, 0, 1600000000)
         a.SetTimeOffset(offset[i], opt)
         offsettimeformat = a.GetTimeFormat() # char
         a_list.append( a )
         
         #buf = "256"
         buf = create_string_buffer(256) 
         if offset[i] < t[i]:
            buf = sprintf(buf, "#splitline{%s, %s}{offset: %ld, option %s}",
               stime(t[i]).Data(), stime(t[i], True).Data(), offset[i], opt)
            
         else:
            h = t[i] / 3600
            m = (t[i] - 3600 * h) / 60
            s = (t[i] - h * 3600 - m * 60)

            buf = sprintf(buf, "#splitline{%d h %d m %d s}{offset: %s, option %s}",
               h, m, s, stime(offset[i], gmt).Data(), opt)
            
         tex1.DrawLatex(.01, .75, buf)
         tex2.DrawLatex(.01, .50, offsettimeformat)

         t_ = t[i] + offset[i]
         buf = sprintf(buf, "Expecting:    #color[2]{%s}", stime(t_, gmt, False).Data())
         tex3.DrawLatex(.01, .24, buf)

         if (i > 0) : l.DrawLine(0, 0.95, 1, 0.95)
         
      
   return c
   


if __name__ == "__main__":
   timeonaxis3()
