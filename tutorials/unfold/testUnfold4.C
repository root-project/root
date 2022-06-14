/// \file
/// \ingroup tutorial_unfold
/// \notebook
/// Test program for the class TUnfoldSys.
///
/// Simple toy tests of the TUnfold package
///
/// Pseudo data (5000 events) are unfolded into three components
/// The unfolding is performed once without and once with area constraint
///
/// Ideally, the pulls may show that the result is biased if no constraint
/// is applied. This is expected because the true data errors are not known,
/// and instead the sqrt(data) errors are used.
///
/// \macro_output
/// \macro_code
///
///  **Version 17.6, in parallel to changes in TUnfold**
///
/// #### History:
///  - Version 17.5, in parallel to changes in TUnfold
///  - Version 17.4, in parallel to changes in TUnfold
///  - Version 17.3, in parallel to changes in TUnfold
///  - Version 17.2, in parallel to changes in TUnfold
///  - Version 17.1, in parallel to changes in TUnfold
///  - Version 16.1, parallel to changes in TUnfold
///  - Version 16.0, parallel to changes in TUnfold
///  - Version 15, use L-curve scan to scan the average correlation
///
///  This file is part of TUnfold.
///
///  TUnfold is free software: you can redistribute it and/or modify
///  it under the terms of the GNU General Public License as published by
///  the Free Software Foundation, either version 3 of the License, or
///  (at your option) any later version.
///
///  TUnfold is distributed in the hope that it will be useful,
///  but WITHOUT ANY WARRANTY; without even the implied warranty of
///  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///  GNU General Public License for more details.
///
///  You should have received a copy of the GNU General Public License
///  along with TUnfold.  If not, see <http://www.gnu.org/licenses/>.
///
/// \author Stefan Schmitt DESY, 14.10.2008

#include <TMath.h>
#include <TCanvas.h>
#include <TRandom3.h>
#include <TFitter.h>
#include <TF1.h>
#include <TStyle.h>
#include <TVector.h>
#include <TGraph.h>
#include <TError.h>

#include "TUnfoldDensity.h"

using namespace std;

TRandom *rnd=0;

Int_t GenerateGenEvent(Int_t nmax,const Double_t *probability) {
   // choose an integer random number in the range [0,nmax]
   //    (the generator level bin)
   // depending on the probabilities
   //   probability[0],probability[1],...probability[nmax-1]
   Double_t f=rnd->Rndm();
   Int_t r=0;
   while((r<nmax)&&(f>=probability[r])) {
      f -= probability[r];
      r++;
   }
   return r;
}

Double_t GenerateRecEvent(const Double_t *shapeParm) {
   // return a coordinate (the reconstructed variable)
   // depending on shapeParm[]
   //  shapeParm[0]: fraction of events with Gaussian distribution
   //  shapeParm[1]: mean of Gaussian
   //  shapeParm[2]: width of Gaussian
   //  (1-shapeParm[0]): fraction of events with flat distribution
   //  shapeParm[3]: minimum of flat component
   //  shapeParm[4]: maximum of flat component
   Double_t f=rnd->Rndm();
   Double_t r;
   if(f<shapeParm[0]) {
      r=rnd->Gaus(shapeParm[1],shapeParm[2]);
   } else {
      r=rnd->Rndm()*(shapeParm[4]-shapeParm[3])+shapeParm[3];
   }
   return r;
}

void testUnfold4(bool printInfo = false)
{

   // switch off printing Info messages
   if (!printInfo) gErrorIgnoreLevel = kWarning; 
   
  // switch on histogram errors
  TH1::SetDefaultSumw2();

  // random generator
  rnd=new TRandom3();

  // data and MC number of events
  Double_t const nData0=    500.0;
  Double_t const nMC0  =  50000.0;

  // Binning
  // reconstructed variable (0-10)
  Int_t const nDet=15;
  Double_t const xminDet=0.0;
  Double_t const xmaxDet=15.0;

  // signal binning (three shapes: 0,1,2)
  Int_t const nGen=3;
  Double_t const xminGen=-0.5;
  Double_t const xmaxGen= 2.5;

  // parameters
  // fraction of events per signal shape
  static const Double_t genFrac[]={0.3,0.6,0.1};

  // signal shapes
  static const Double_t genShape[][5]=
     {{1.0,2.0,1.5,0.,15.},
      {1.0,7.0,2.5,0.,15.},
      {0.0,0.0,0.0,0.,15.}};

  // define DATA histograms
  // observed data distribution
  TH1D *histDetDATA=new TH1D("Yrec",";DATA(Yrec)",nDet,xminDet,xmaxDet);

  // define MC histograms
  // matrix of migrations
  TH2D *histGenDetMC=new TH2D("Yrec%Xgen","MC(Xgen,Yrec)",
                              nGen,xminGen,xmaxGen,nDet,xminDet,xmaxDet);

  TH1D *histUnfold=new TH1D("Xgen",";DATA(Xgen)",nGen,xminGen,xmaxGen);

  TH1D **histPullNC=new TH1D* [nGen];
  TH1D **histPullArea=new TH1D* [nGen];
  for(int i=0;i<nGen;i++) {
     histPullNC[i]=new TH1D(TString::Format("PullNC%d",i),"pull",15,-3.,3.);
     histPullArea[i]=new TH1D(TString::Format("PullArea%d",i),"pull",15,-3.,3.);
  }

  // this method is new in version 16 of TUnfold
  cout<<"TUnfold version is "<<TUnfold::GetTUnfoldVersion()<<"\n";

  for(int itoy=0;itoy<1000;itoy++) {
     if(!(itoy %10)) cout<<"toy iteration: "<<itoy<<"\n";
     histDetDATA->Reset();
     histGenDetMC->Reset();

     Int_t nData=rnd->Poisson(nData0);
     for(Int_t i=0;i<nData;i++) {
        Int_t iGen=GenerateGenEvent(nGen,genFrac);
        Double_t yObs=GenerateRecEvent(genShape[iGen]);
        histDetDATA->Fill(yObs);
     }

     Int_t nMC=rnd->Poisson(nMC0);
     for(Int_t i=0;i<nMC;i++) {
        Int_t iGen=GenerateGenEvent(nGen,genFrac);
        Double_t yObs=GenerateRecEvent(genShape[iGen]);
        histGenDetMC->Fill(iGen,yObs);
     }
     /* for(Int_t ix=0;ix<=histGenDetMC->GetNbinsX()+1;ix++) {
        for(Int_t iy=0;iy<=histGenDetMC->GetNbinsY()+1;iy++) {
           cout<<ix<<iy<<" : "<<histGenDetMC->GetBinContent(ix,iy)<<"\n";
        }
        } */
     //========================
     // unfolding

     TUnfoldSys unfold(histGenDetMC,TUnfold::kHistMapOutputHoriz,
                       TUnfold::kRegModeSize,TUnfold::kEConstraintNone);
     // define the input vector (the measured data distribution)
     unfold.SetInput(histDetDATA,0.0,1.0);

     // run the unfolding
     unfold.ScanLcurve(50,0.,0.,0,0,0);

     // fill pull distributions without constraint
     unfold.GetOutput(histUnfold);

     for(int i=0;i<nGen;i++) {
        histPullNC[i]->Fill((histUnfold->GetBinContent(i+1)-genFrac[i]*nData0)/
                            histUnfold->GetBinError(i+1));
     }

     // repeat unfolding on the same data, now with Area constraint
     unfold.SetConstraint(TUnfold::kEConstraintArea);

     // run the unfolding
     unfold.ScanLcurve(50,0.,0.,0,0,0);

     // fill pull distributions with constraint
     unfold.GetOutput(histUnfold);

     for(int i=0;i<nGen;i++) {
        histPullArea[i]->Fill((histUnfold->GetBinContent(i+1)-genFrac[i]*nData0)/
                            histUnfold->GetBinError(i+1));
     }

  }
  TCanvas output;
  output.Divide(3,2);

  gStyle->SetOptFit(1111);

  for(int i=0;i<nGen;i++) {
     output.cd(i+1);
     histPullNC[i]->Fit("gaus");
     histPullNC[i]->Draw();
  }
  for(int i=0;i<nGen;i++) {
     output.cd(i+4);
     histPullArea[i]->Fit("gaus");
     histPullArea[i]->Draw();
  }
  output.SaveAs("testUnfold4.ps");
}
