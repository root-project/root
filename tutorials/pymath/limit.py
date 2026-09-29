## \file
## \ingroup tutorial_math
## \notebooks
## This program demostrates how to compute a probability upto 
## a 95 % Confidence Interval(C.L.) with user defined limits.
## It uses a set of randomly created histograms to achieve that.
## Enjoy!
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Christophe Delaere
## \translator P. P.


import ROOT

iostream = ROOT.iostream 
TH1 = ROOT.TH1 
THStack = ROOT.THStack 
TCanvas = ROOT.TCanvas 
TFrame = ROOT.TFrame 
TRandom2 = ROOT.TRandom2 
TSystem = ROOT.TSystem 
TVector = ROOT.TVector 
TObjArray = ROOT.TObjArray 
TLimit = ROOT.TLimit 
TLimitDataSource = ROOT.TLimitDataSource 
TConfidenceLevel = ROOT.TConfidenceLevel 

#
TH1D = ROOT.TH1D
TVectorD = ROOT.TVectorD
TObjString = ROOT.TObjString

#constants
kBlue = ROOT.kBlue

#types
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t


# void
def limit() :
   # Create a new canvas.
   global c1
   c1 = TCanvas("c1","Dynamic Filling Example",200,10,700,500)
   c1.SetFillColor(42)
   
   # Create some histograms.
   global backgroundHist, signalHist, dataHist
   backgroundHist = TH1D("background","The expected background",30,-4,4)
   signalHist = TH1D("signal","the expected signal",30,-4,4)
   dataHist = TH1D("data","some fake data points",30,-4,4)

   # Setting-up historgrams.
   backgroundHist.SetFillColor(48)
   signalHist.SetFillColor(41)
   dataHist.SetMarkerStyle(21)
   dataHist.SetMarkerColor(kBlue)
   backgroundHist.Sumw2(); # needed for stat uncertainty
   signalHist.Sumw2(); # needed for stat uncertainty
   
   # Fill histograms randomly
   r = TRandom2()
   bg , sig , dt = [ Float_t() for _ in range(3) ] #
   #for (Int_t i = 0; i < 25000; i++) {
   for i in range(0, 25000, 1): 
      bg = r.Gaus(0,1)
      sig = r.Gaus(1,.2)
      backgroundHist.Fill(bg,0.02)
      signalHist.Fill(sig,0.001)
      
   #for(Int_t i = 0; i < 500; i++) {
   for i in range(0, 500, 1):
      dt = r.Gaus(0,1)
      dataHist.Fill(dt)
      
   global hs
   hs = THStack("hs","Signal and background compared to data...")
   hs.Add(backgroundHist)
   hs.Add(signalHist)
   hs.Draw("hist")
   dataHist.Draw("PE1,Same")
   c1.Modified()
   c1.Update()
   c1.GetFrame().SetFillColor(21)
   c1.GetFrame().SetBorderSize(6)
   c1.GetFrame().SetBorderMode(-1)
   c1.Modified()
   c1.Update()
   ROOT.gSystem.ProcessEvents()
   
   # Compute the limits
   print(f"Computing limits... ")
   global mydatasource, myconfidence
   mydatasource = TLimitDataSource(signalHist,backgroundHist,dataHist)
   myconfidence = TLimit.ComputeLimit(mydatasource,50000)
   print(f"CLs    : "   , myconfidence.CLs() )
   print(f"CLsb   : "   , myconfidence.CLsb())
   print(f"CLb    : "   , myconfidence.CLb() )
   print(f"< CLs >  : " , myconfidence.GetExpectedCLs_b() )
   print(f"< CLsb > : " , myconfidence.GetExpectedCLsb_b())
   print(f"< CLb >  : " , myconfidence.GetExpectedCLb_b() )
   
   # Add stat uncertainty
   print(f"\n Computing limits with stat systematics... ")
   mystatconfidence = TLimit.ComputeLimit(mydatasource,50000,True)
   print(f"CLs    : "   , mystatconfidence.CLs() )
   print(f"CLsb   : "   , mystatconfidence.CLsb())
   print(f"CLb    : "   , mystatconfidence.CLb() )
   print(f"< CLs >  : " , mystatconfidence.GetExpectedCLs_b() )
   print(f"< CLsb > : " , mystatconfidence.GetExpectedCLsb_b())
   print(f"< CLb >  : " , mystatconfidence.GetExpectedCLb_b() )
   
   # Add some systematics
   print("\nComputing limits with systematics... ")
   global errorb, errors, names
   errorb = TVectorD(2)
   errors = TVectorD(2)
   names = TObjArray()

   name1 = TObjString("bg uncertainty")
   name2 = TObjString("sig uncertainty")
   names.AddLast(name1)
   names.AddLast(name2)
   errorb[0]=0.05; # error source 1: 5%
   errorb[1]=0;    # error source 2: 0%
   errors[0]=0;    # error source 1: 0%
   errors[1]=0.01; # error source 2: 1%

   global mynewdatasource
   mynewdatasource = TLimitDataSource()
   mynewdatasource.AddChannel(signalHist,backgroundHist,dataHist,errors,errorb,names)
   mynewconfidence = TLimit.ComputeLimit(mynewdatasource,50000,True)
   print(f"CLs    : " , mynewconfidence.CLs() )
   print(f"CLsb   : " , mynewconfidence.CLsb())
   print(f"CLb    : " , mynewconfidence.CLb() )
   print(f"< CLs >  : " , mynewconfidence.GetExpectedCLs_b() )
   print(f"< CLsb > : " , mynewconfidence.GetExpectedCLsb_b())
   print(f"< CLb >  : " , mynewconfidence.GetExpectedCLb_b() )
   
   # show canonical -2lnQ plots in a new canvas
   # - The histogram of -2lnQ for background hypothesis (full)
   # - The histogram of -2lnQ for signal and background hypothesis (dashed)
   global c2
   c2 = TCanvas("c2")
   myconfidence.Draw()
   
   # clean up (except histograms and canvas)
   del myconfidence
   del mydatasource
   del mystatconfidence
   del mynewconfidence
   del mynewdatasource
   


if __name__ == "__main__":
   limit()
