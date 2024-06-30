## \file
## \ingroup tutorial_math
## \notebook
##
## Tutorial illustrating how to create a plot by comparing a Breit Wigner to a Relativistic Breit Wigner.
##
## can be run with:
##
## ~~~{.py}
##  IPython[0] %run BreitWigner.py
## ~~~
##
## \macro_image
## \macro_code
##
## \author Jack Lindon
## \translator P. P.


import ROOT
import ctypes

TMath = ROOT.TMath 
#limits = ROOT.limits # ROOT doesn't have <limits>
string = ROOT.string 
TAxis = ROOT.TAxis 
TGraph = ROOT.TGraph 
TCanvas = ROOT.TCanvas 
TLatex = ROOT.TLatex 
TLegend = ROOT.TLegend 
#For gStyle to remove stat box. 
TStyle = ROOT.TStyle 

#constants
kBlue = ROOT.kBlue
kBlack = ROOT.kBlack

#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t
c_double = ctypes.c_double

#std
std = ROOT.std

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad



#utils
def to_c(ls):
   return (c_double * len(ls))(*ls)

#void
def plotTwoTGraphs(x : [Double_t], y1 : [Double_t], y2 : [Double_t]
                 , nPoints : Int_t
                 , lowerXLimit : Double_t, upperXLimit : Double_t
                 , lowerYLimit : Double_t, upperYLimit : Double_t
                 , legend1 : string, legend2 : string
                 , plotTitle1 : string, plotTitle2 : string, plotTitle3 : string
                 , pdfTitle : string) : 
   
   ####################################
   #Define variables for plot aesthetics and positioning
   legendXPos = 0.63
   legendYPos = 0.85
   legendXWidth = 0.29
   legendYHeight = 0.1
   plotTitleXPos = 0.23
   plotTitleYPos = 0.25
   fontSize = 0.04
   lineWidth = 2
   setLimitPlotLogScale = True
   xAxisTitle = "E [GeV]"
   yAxisTitle = "Events"
   xAxisTitleOffset = 1
   yAxisTitleOffset = 1.3
   gStyle.SetOptStat(0)
   
   ####################################
   # Initialize TGraphs
   global gr1, gr2
   gr1 = TGraph(nPoints,x,y1)
   gr2 = TGraph(nPoints,x,y2)
   gr1.SetLineWidth(lineWidth)
   gr2.SetLineWidth(lineWidth)
   gr1.SetLineColor(kBlack)
   gr2.SetLineColor(kBlue)
   
   ######################################
   # Initialize canvas
   global c1
   c1 = TCanvas("c1","transparent pad",200,10,600,600)
   c1.SetLogy(setLimitPlotLogScale)
   c1.SetTicks(1,1)
   c1.SetRightMargin(0.02)
   c1.SetTopMargin(0.02)
   
   
   ####################################
   #Make just a basic invisible TGraph just for the axes
   axis_x = [lowerXLimit, upperXLimit]
   axis_y = [lowerYLimit, upperYLimit]
   #Convertion to C types
   axis_x = to_c(axis_x)
   axis_y = to_c(axis_y)

   global grAxis
   grAxis = TGraph(2, axis_x, axis_y)
   grAxis.SetTitle("")
   grAxis.GetYaxis().SetTitle(str(yAxisTitle))
   grAxis.GetXaxis().SetTitle(str(xAxisTitle))
   grAxis.GetXaxis().SetRangeUser(lowerXLimit,upperXLimit)
   grAxis.GetYaxis().SetRangeUser(lowerYLimit,upperYLimit)
   grAxis.GetXaxis().SetLabelSize(fontSize)
   grAxis.GetYaxis().SetLabelSize(fontSize)
   grAxis.GetXaxis().SetTitleSize(fontSize)
   grAxis.GetYaxis().SetTitleSize(fontSize)
   grAxis.GetXaxis().SetTitleOffset(xAxisTitleOffset)
   grAxis.GetYaxis().SetTitleOffset(yAxisTitleOffset)
   grAxis.SetLineWidth(0);#So invisible
   
   #######################################
   # Make legend and set aesthetics
   global legend
   legend = TLegend(legendXPos, legendYPos, legendXPos + legendXWidth, legendYPos + legendYHeight)
   legend.SetFillStyle(0)
   legend.SetBorderSize(0)
   legend.SetTextSize(fontSize)
   legend.AddEntry(gr1,str(legend1),"L")
   legend.AddEntry(gr2,str(legend2),"L")
   
   
   ########################################
   # Add plot title to plot. Make in three lines so not crowded.
   # Shift each line down by shiftY
   shiftY = 0.037 # float 
   global tex_Title, tex_Title2, tex_Title3
   tex_Title = TLatex(plotTitleXPos,plotTitleYPos-0*shiftY,str(plotTitle1))
   tex_Title.SetNDC()
   tex_Title.SetTextFont(42)
   tex_Title.SetTextSize(fontSize)
   tex_Title.SetLineWidth(lineWidth)
   tex_Title2 = TLatex(plotTitleXPos,plotTitleYPos-1*shiftY,str(plotTitle2))
   tex_Title2.SetNDC()
   tex_Title2.SetTextFont(42)
   tex_Title2.SetTextSize(fontSize)
   tex_Title2.SetLineWidth(lineWidth)
   tex_Title3 = TLatex(plotTitleXPos,plotTitleYPos-2*shiftY,str(plotTitle3))
   tex_Title3.SetNDC()
   tex_Title3.SetTextFont(42)
   tex_Title3.SetTextSize(fontSize)
   tex_Title3.SetLineWidth(lineWidth)
   
   
   ########################
   # Draw everything
   grAxis.Draw("AL")
   gr1.Draw("L same")
   gr2.Draw("L same")
   legend.Draw()
   tex_Title.Draw()
   tex_Title2.Draw()
   tex_Title3.Draw()
   c1.RedrawAxis(); #Be sure to redraw axis AFTER plotting TGraphs otherwise TGraphs will be on top of tick marks and axis borders.
   
   gPad.Print(str(pdfTitle))
   
   


# void
def BreitWigner() :
   
   ######################################
   # Define x axis limits and steps for each plotted point
   nPoints = 1000
   xMinimum = 0
   xMaximum = 13000
   xStepSize = (xMaximum-xMinimum)/nPoints
   
   ####################################
   # Define arrays of (x,y) points.
   x =          [ Double_t() ] * nPoints # Double_t[] 
   y_nonRelBW = [ Double_t() ] * nPoints # Double_t[]
   y_relBW =    [ Double_t() ] * nPoints # Double_t[]
   # To C type
   x = to_c(x)
   y_nonRelBW = to_c(y_nonRelBW)
   y_relBW = to_c(y_relBW)
   
   
   ######################
   # Define Breit-Wigner parameters
   width = 1350
   sigma = 269.7899
   median = 9000
   
   ##################################
   # Loop over x axis range, filling in (x,y) points,
   # and finding y minimums and maximums for axis limit.
   yMinimum = std.numeric_limits["Double_t"].max()
   # Note: 
   # >>>>>   y-maximum is at x = median 
   #       (
   #       and non-relativistic IS relativistic at x = median; 
   #       so, the choice of whatever function does not matter.
   #       )
   yMaximum = TMath.BreitWignerRelativistic(median,median,width); 
   
   #for (Int_t i=0; i<nPoints; i++) {
   for i in range(0, nPoints, 1):
      currentX = xMinimum + i * xStepSize
      x[i] = currentX
      y_nonRelBW[i] = TMath.BreitWigner(currentX, median, width)
      y_relBW[i]    = TMath.BreitWignerRelativistic(currentX, median, width)
      
      if y_nonRelBW[i]<yMinimum:
         yMinimum = y_nonRelBW[i]
         
      if y_relBW[i]<yMinimum:
         yMinimum = y_relBW[i]
         
      
   # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
   # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
   plotTwoTGraphs(
   x, y_nonRelBW, y_relBW
   , nPoints
   , xMinimum, xMaximum #xAxis limits
   , yMinimum/4, yMaximum*4 #yAxis limits, expand for aesthetics.
   ,"NonRel BW", "Rel BW" #Legend entries
   , "Comparing BW", "M = " + std.to_string(int(round(median))) + " GeV"
   , "#Gamma = " + std.to_string(int(round(width))) + " GeV" #Plot Title entry (three lines)
   , "BW_M"+std.to_string(int(round(median)))+"_Gamma" + std.to_string(int(round(width))) +".pdf)" #PDF file title.
   )
   


if __name__ == "__main__":
   BreitWigner()
