# \file
# \ingroup tutorial_math
# \notebook
# Example of Chebyshev polynomials
# Using TFormula and pre-defined functions of Checpolynomials.
#
# \macro_image
# \macro_code
#
# \author Lorenzo Moneta

import ROOT

#classes
TLegend = ROOT.TLegend
TF1 = ROOT.TF1
TString = ROOT.TString

#colors
kRed = ROOT.kRed
kMagenta = ROOT.kMagenta
kBlue = ROOT.kBlue
kCyan = ROOT.kCyan
kGreen = ROOT.kGreen
kYellow = ROOT.kYellow
kOrange = ROOT.kOrange



def ChebyshevPol():
   
   
   global legend
   legend = TLegend(0.88,0.4,1.,1.)
   
   # adding 3 helps changes a bit its reference-color.
   # helps to visualize better many functions in one canvas.
   colors = [ kRed, kRed+3, kMagenta, kMagenta+3, kBlue, kBlue+3, kCyan+3, kGreen, kGreen+3, kYellow, kOrange ]

   global functions
   functions = []  
   for degree in range(0, 10+1, 1):
      title = TString.Format( "cheb%d"%degree)

      functions.append( 
         TF1("f1", str(title),-1,1) 
      )

      # all parameters are zero except the one corresponding to the degree
      functions[degree].SetParameter(degree,1)

      # setting-up 
      functions[degree].SetLineColor( colors[degree])
      functions[degree].SetMinimum(-1.2)
      functions[degree].SetMaximum(1.2)
      opt = "" if (degree == 0) else "same"
      #functions[degree].Print("V")
      functions[degree].SetNpx(1000)
      functions[degree].SetTitle("Chebyshev Polynomial")
      functions[degree].Draw(opt)
      
      # adding legends
      number = TString.Format( "N=%d"%degree )
      legend.AddEntry(functions[degree],str(number) ,"L")

      
   legend.Draw()
   
if __name__ == "__main__":
   ChebyshevPol()
