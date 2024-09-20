## \file
## \ingroup tutorial_graphs
## \notebook -js
##
## This script illustrates how to use a time mode on the axis
## with different time intervals-and-formats.
## Through all this script, the time is expressed in UTC format. Some
## information about it(and others like GPS-format) can be found at
## <a href="http:#tycho.usno.navy.mil/systime.html">http:#tycho.usno.navy.mil/systime.html</a>
##  or
## <a href="http:#www.topology.org/sci/time.html">http:#www.topology.org/sci/time.html</a>
##
## The start time is : almost "now" actually; because the time at which the script is executed is
## retarded from the time you will see its print on prompt-terminal.
## Actually, the nearest preceding hour since the beginning of this script.
## The time is in general expressed in UTC-time with the C-time() function.
## This will obviously generally not be the time displayed on your watch( or your current time),
## since it is a universal time(no an universal time or just universal time, look it up in grammar). 
## See the C-time functions for converting this time
## into more useful structures.
##
## Enjoy timing!
##
## \macro_image
## \macro_code
##
## \author Damir Buskulic
## \translator P. P.


import ROOT
import ctypes
#Increase maxdigits from 4300 to 4300000. Some integer values are too huge.
#TODO: Divide long integers value into separate integers.
import sys
sys.set_int_max_str_digits(maxdigits = 4300000)

#standard library
std = ROOT.std
time = std.time

#classes
TCanvas = ROOT.TCanvas
TH1F = ROOT.TH1F
TGraph = ROOT.TGraph

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
exp = ROOT.exp

#types
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
nullptr = ROOT.nullptr
c_double = ctypes.c_double
c_uint = ctypes.c_uint
c_long = ctypes.c_long
c_longdouble = ctypes.c_longdouble
time_t = std.time_t


#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def to_c_uint( ls ):
   return (c_uint * len(ls) )( * ls )
def to_c_long( ls ):
   return (c_long * len(ls) )( * ls )
def to_c_longdouble( ls ):
   return (c_longdouble * len(ls) )( * ls )
def printf(string, *args):
   print( string % args )
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

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




# TCanvas
def timeonaxis() :
   
   global script_time
   script_time = time_t()

   script_time = time(0)
   script_time = 3600 * Int_t(script_time/3600)
   
   # The offset-time is the one that will be used by all graphs.
   # If one changes it, it will be changed even on the graphs already defined.
   # Be cautious.
   gStyle.SetTimeOffset(script_time)
   

   global ct
   ct = TCanvas("ct","Time on axis",10,10,700,900)
   ct.Divide(1,3)
   
   
   # I Part.
   #### Build a signal : noisy damped sine
   #        Time interval : 30 minutes
  
   #Setting-up style and format. 
   gStyle.SetTitleH(0.08)

   global noise
   noise = Float_t()

   global ht
   ht = TH1F("ht","Love at first sight",3000,0.,2000.)
   #for (i=1; i<3000; i++) {
   for i in range(1, 3000, 1):
      noise = gRandom.Gaus(0,120)
      if i>700:
         noise += 1000*sin((i-700)*6.28/30)*exp((700-i)/300.)
         
      ht.SetBinContent(i,noise)
      
   ct.cd(1)

   ht.SetLineColor(2)
   ht.GetXaxis().SetLabelSize(0.05)

   # Sets time on the X axis.
   ht.GetXaxis().SetTimeDisplay(1)
   # Note:
   # The used-time is the one that sets as an offset-time which is added to axis value.
   # Then, this is converted into a "day/month/year hour:min:sec" formata, and
   # a "reasonable" tick interval value is chosen. See below to unfold what do I mean by "reasonable".
   ht.Draw()
   


   # II Part.
   #### Build a simple graph beginning at a different time from the time being.
   #        Time interval : 5 seconds
   
   x = [ Float_t() for i in range(100) ] # Float_t * [100] 
   t = [ Float_t() for i in range(100) ] # Float_t * [100] 

   #for (i=0; i<100; i++) {
   for i in range(0, 100, 1):
      x[i] = sin(i*4*3.1415926/50)*exp(-i/20.)
      t[i] = 6000+i/20.
      
   ct.cd(2)

   c_x = to_c( x ) 
   c_t = to_c( t ) 

   global gt
   gt = TGraph(100,c_t,c_x)
   gt.SetTitle("Politics")
   gt.SetLineColor(5)
   gt.SetLineWidth(2)
   gt.GetXaxis().SetLabelSize(0.05)

   # Sets time on the X axis
   gt.GetXaxis().SetTimeDisplay(1)

   gt.Draw("AL")
   gPad.Modified()
   


   # III Part.
   #### Build a second graph for intervals of long time, like years.
   #        Time interval : a few years
   
   x2 = [ Float_t() for i in range(100) ] # Float_t * [100] 
   t2 = [ Float_t() for i in range(100) ] # Float_t * [100] 
   
   #for (i=0; i<10; i++) {
   for i in range(1, 10, 1):
      x2[i] = gRandom.Gaus(500,100)*i
      t2[i] = i*365*86400
      #Debug: print(" x[i] ",  x2[i], "    t[i]  ", t2[i])
      
   ct.cd(3)

   ##Debug:
   #global gt2
   #gt2 = t2 
   #import sys
   #sys.exit()
   #Note:
   #Error:
   #ValueError: Exceeds the limit (4300 digits) for integer string conversion; 
   #use sys.set_int_max_str_digits() to increase the limit
   #  
   #c_t = to_c_longdouble( t ) 
   #c_x = to_c_longdouble( x ) 

   c_t2 = to_c( t2 )
   c_x2 = to_c( x2 )
   global gt2
   gt2 = TGraph(10,c_t2,c_x2)
   gt2.SetTitle("Number of monkeys on the moon")
   gt2.SetMarkerColor(4)
   gt2.SetMarkerStyle(29)
   gt2.SetMarkerSize(1.3)
   gt2.GetXaxis().SetLabelSize(0.05)

   # Sets time on the X axis
   gt2.GetXaxis().SetTimeDisplay(1)
   gt2.GetXaxis().SetTimeFormat("y. %Y %F2000-01-01 00:00:00")

   gt2.Draw("AP")
   gPad.Modified()

   return ct


   #Note:
   # For your information, if you are interested in C-time formats.
   # Specifically for this line above:
   # `t2.GetXaxis().SetTimeFormat("y. %Y %F2000-01-01 00:00:00")`
   # One can choose a different time format than the one chosen by default.
   # Let's say, the time format is the same as the one by the C-strftime() function;
   # and it is a string containing the next formats :
   #
   #    for date :
   #      %a abbreviated weekday name
   #      %b abbreviated month name
   #      %d day of the month (01-31)
   #      %m month (01-12)
   #      %y year without century
   #      %Y year with century
   #
   #    for time :
   #      %H hour (24-hour clock)
   #      %I hour (12-hour clock)
   #      %p local equivalent of AM or PM
   #      %M minute (00-59)
   #      %S seconds (00-61)
   #      %% %
   # The other output-characters are similar. 
   
   


if __name__ == "__main__":
   timeonaxis()
