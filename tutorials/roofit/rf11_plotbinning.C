/////////////////////////////////////////////////////////////////////////
//
// RooFit tutorial macro #11
// 
// Plotting unbinned data with alternate and variable binnings
//
// 
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussModel.h"
#include "RooDecay.h"
#include "RooBMixDecay.h"
#include "RooCategory.h"
#include "RooBinning.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;


void rf11_plotbinning()
{

  // Build a B decay p.d.f with mixing
  RooRealVar dt("dt","dt",-20,20) ;
  RooRealVar dm("dm","dm",0.472) ;
  RooRealVar tau("tau","tau",1.547) ;
  RooRealVar w("w","mistag rate",0.1) ;
  RooRealVar dw("dw","delta mistag rate",0.) ;

  RooCategory mixState("mixState","B0/B0bar mixing state") ;
  mixState.defineType("mixed",-1) ;
  mixState.defineType("unmixed",1) ;
  RooCategory tagFlav("tagFlav","Flavour of the tagged B0") ;
  tagFlav.defineType("B0",1) ;
  tagFlav.defineType("B0bar",-1) ;

  // Build a gaussian resolution model
  RooRealVar dterr("dterr","dterr",0.1,1.0) ;
  RooRealVar bias1("bias1","bias1",0) ;
  RooRealVar sigma1("sigma1","sigma1",0.1) ;  
  RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1) ;

  // Construct Bdecay (x) gauss
  RooBMixDecay bmix("bmix","decay",dt,mixState,tagFlav,tau,dm,w,dw,gm1,RooBMixDecay::DoubleSided) ;

  // Sample 2000 events in (dt,mixState,tagFlav) from bmix
  RooDataSet *data = bmix.generate(RooArgSet(dt,mixState,tagFlav),2000) ;



  // --- Make plot of dt distribution of data in range (-15,15)
  //     with fine binning for dt>0 and coarse binning for dt<0

  // Create binning object with range (-15,15)
  RooBinning tbins(-15,15) ;

  // Add 60 bins with uniform spacing in range (-15,0)
  tbins.addUniform(60,-15,0) ;

  // Add 15 bins with uniform spacing in range (0,15)
  tbins.addUniform(15,0,15) ;
  
  // Make plot with specified binning
  RooPlot* dtframe = dt.frame(-15,15) ;
  data->plotOn(dtframe,Binning(tbins)) ;
  bmix.plotOn(dtframe) ;
  
  
  // --- Make plot of dt distribution of data asymmetry in 'mixState'
  //     with variable binning 

  // Create binning object with range (-10,10)
  RooBinning abins(-10,10) ;

  // Add boundaries at 0, (-1,1), (-2,2), (-3,3), (-4,4) and (-6,6)
  abins.addBoundary(0) ;
  abins.addBoundaryPair(1) ;
  abins.addBoundaryPair(2) ;
  abins.addBoundaryPair(3) ;
  abins.addBoundaryPair(4) ;
  abins.addBoundaryPair(6) ;
  
  // Create plot frame in dt
  RooPlot* aframe = dt.frame(-10,10) ;

  // Plot mixState asymmetry of data with specified customg binning
  data->plotOn(aframe,Asymmetry(mixState),Binning(abins)) ;

  // Plot corresponding property of p.d.f
  bmix.plotOn(aframe,Asymmetry(mixState)) ;

  // Adjust vertical range of plot to sensible values for an asymmetry
  aframe->SetMinimum(-1.1) ;
  aframe->SetMaximum(1.1) ;



  // Draw plots on canvas
  TCanvas* c = new TCanvas("rf10_ranges","rf10_ranges",800,400) ;
  c->Divide(2) ;
  c->cd(1) ; dtframe->Draw() ;
  c->cd(2) ; aframe->Draw() ;
  
}
