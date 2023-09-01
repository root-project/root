/// \file
/// \ingroup tutorial_unfold
/// \notebook
/// Test program for the classes TUnfoldDensity and TUnfoldBinning
///
/// A toy test of the TUnfold package
///
/// This is an example of unfolding a two-dimensional distribution
/// also using an auxiliary measurement to constrain some background
///
/// The example comprises several macros
///  - testUnfold5a.C   create root files with TTree objects for
///                      signal, background and data
///            - write files  testUnfold5_signal.root
///                            testUnfold5_background.root
///                            testUnfold5_data.root
///
///  - testUnfold5b.C   create a root file with the TUnfoldBinning objects
///            - write file  testUnfold5_binning.root
///
///  - testUnfold5c.C   loop over trees and fill histograms based on the
///                      TUnfoldBinning objects
///            - read  testUnfold5_binning.root
///                     testUnfold5_signal.root
///                     testUnfold5_background.root
///                     testUnfold5_data.root
///
///            - write testUnfold5_histograms.root
///
///  - testUnfold5d.C   run the unfolding
///            - read  testUnfold5_histograms.root
///            - write testUnfold5_result.root
///                     testUnfold5_result.ps
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
///  - Version 17.0 example for multi-dimensional unfolding
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

#include <iostream>
#include <map>
#include <cmath>
#include <TMath.h>
#include <TRandom3.h>
#include <TFile.h>
#include <TTree.h>

using namespace std;

TRandom *g_rnd=0;

class ToyEvent {
public:
   void GenerateDataEvent(TRandom *rnd);
   void GenerateSignalEvent(TRandom *rnd);
   void GenerateBgrEvent(TRandom *rnd);
   // reconstructed quantities
   inline Double_t GetPtRec(void) const { return fPtRec; }
   inline Double_t GetEtaRec(void) const { return fEtaRec; }
   inline Double_t GetDiscriminator(void) const {return fDiscriminator; }
   inline Bool_t IsTriggered(void) const { return fIsTriggered; }

   // generator level quantities
   inline Double_t GetPtGen(void) const {
      if(IsSignal()) return fPtGen;
      else return -1.0;
   }
   inline Double_t GetEtaGen(void) const {
       if(IsSignal()) return fEtaGen;
       else return 999.0;
   }
   inline Bool_t IsSignal(void) const { return fIsSignal; }
protected:

   void GenerateSignalKinematics(TRandom *rnd,Bool_t isData);
   void GenerateBgrKinematics(TRandom *rnd,Bool_t isData);
   void GenerateReco(TRandom *rnd);

   // reconstructed quantities
   Double_t fPtRec;
   Double_t fEtaRec;
   Double_t fDiscriminator;
   Bool_t fIsTriggered;
   // generated quantities
   Double_t fPtGen;
   Double_t fEtaGen;
   Bool_t fIsSignal;

   static Double_t kDataSignalFraction;

};

void testUnfold5a()
{
  // random generator
  g_rnd=new TRandom3();

  // data and MC number of events
  const Int_t neventData         =  20000;
  Double_t const neventSignalMC  =2000000;
  Double_t const neventBgrMC     =2000000;

  Float_t etaRec,ptRec,discr,etaGen,ptGen;
  Int_t istriggered,issignal;

  //==================================================================
  // Step 1: generate data TTree

  TFile *dataFile=new TFile("testUnfold5_data.root","recreate");
  TTree *dataTree=new TTree("data","event");

  dataTree->Branch("etarec",&etaRec,"etarec/F");
  dataTree->Branch("ptrec",&ptRec,"ptrec/F");
  dataTree->Branch("discr",&discr,"discr/F");

  // for real data, only the triggered events are available
  dataTree->Branch("istriggered",&istriggered,"istriggered/I");
  // data truth parameters
  dataTree->Branch("etagen",&etaGen,"etagen/F");
  dataTree->Branch("ptgen",&ptGen,"ptgen/F");
  dataTree->Branch("issignal",&issignal,"issignal/I");

  cout<<"fill data tree\n";

  Int_t nEvent=0,nTriggered=0;
  while(nTriggered<neventData) {
     ToyEvent event;
     event.GenerateDataEvent(g_rnd);

     etaRec=event.GetEtaRec();
     ptRec=event.GetPtRec();
     discr=event.GetDiscriminator();
     istriggered=event.IsTriggered() ? 1 : 0;
     etaGen=event.GetEtaGen();
     ptGen=event.GetPtGen();
     issignal=event.IsSignal() ? 1 : 0;

     dataTree->Fill();

     if(!(nEvent%100000)) cout<<"   data event "<<nEvent<<"\n";

     if(istriggered) nTriggered++;
     nEvent++;

  }

  dataTree->Write();
  delete dataTree;
  delete dataFile;

  //==================================================================
  // Step 2: generate signal TTree

  TFile *signalFile=new TFile("testUnfold5_signal.root","recreate");
  TTree *signalTree=new TTree("signal","event");

  signalTree->Branch("etarec",&etaRec,"etarec/F");
  signalTree->Branch("ptrec",&ptRec,"ptrec/F");
  signalTree->Branch("discr",&discr,"discr/F");
  signalTree->Branch("istriggered",&istriggered,"istriggered/I");
  signalTree->Branch("etagen",&etaGen,"etagen/F");
  signalTree->Branch("ptgen",&ptGen,"ptgen/F");

  cout<<"fill signal tree\n";

  for(int ievent=0;ievent<neventSignalMC;ievent++) {
     ToyEvent event;
     event.GenerateSignalEvent(g_rnd);

     etaRec=event.GetEtaRec();
     ptRec=event.GetPtRec();
     discr=event.GetDiscriminator();
     istriggered=event.IsTriggered() ? 1 : 0;
     etaGen=event.GetEtaGen();
     ptGen=event.GetPtGen();

     if(!(ievent%100000)) cout<<"   signal event "<<ievent<<"\n";

     signalTree->Fill();
  }

  signalTree->Write();
  delete signalTree;
  delete signalFile;

  // ==============================================================
  // Step 3: generate background MC TTree

  TFile *bgrFile=new TFile("testUnfold5_background.root","recreate");
  TTree *bgrTree=new TTree("background","event");

  bgrTree->Branch("etarec",&etaRec,"etarec/F");
  bgrTree->Branch("ptrec",&ptRec,"ptrec/F");
  bgrTree->Branch("discr",&discr,"discr/F");
  bgrTree->Branch("istriggered",&istriggered,"istriggered/I");

  cout<<"fill background tree\n";

  for(int ievent=0;ievent<neventBgrMC;ievent++) {
     ToyEvent event;
     event.GenerateBgrEvent(g_rnd);
     etaRec=event.GetEtaRec();
     ptRec=event.GetPtRec();
     discr=event.GetDiscriminator();
     istriggered=event.IsTriggered() ? 1 : 0;

     if(!(ievent%100000)) cout<<"   background event "<<ievent<<"\n";

     bgrTree->Fill();
  }

  bgrTree->Write();
  delete bgrTree;
  delete bgrFile;
}

Double_t ToyEvent::kDataSignalFraction=0.8;

void ToyEvent::GenerateDataEvent(TRandom *rnd) {
   fIsSignal=rnd->Uniform()<kDataSignalFraction;
   if(IsSignal()) {
      GenerateSignalKinematics(rnd,kTRUE);
   } else {
      GenerateBgrKinematics(rnd,kTRUE);
   }
   GenerateReco(rnd);
}

void ToyEvent::GenerateSignalEvent(TRandom *rnd) {
   fIsSignal=1;
   GenerateSignalKinematics(rnd,kFALSE);
   GenerateReco(rnd);
}

void ToyEvent::GenerateBgrEvent(TRandom *rnd) {
   fIsSignal=0;
   GenerateBgrKinematics(rnd,kFALSE);
   GenerateReco(rnd);
}

void ToyEvent::GenerateSignalKinematics(TRandom *rnd,Bool_t isData) {
   Double_t e_T0=0.5;
   Double_t e_T0_eta=0.0;
   Double_t e_n=2.0;
   Double_t e_n_eta=0.0;
   Double_t eta_p2=0.0;
   Double_t etaMax=4.0;
   if(isData) {
      e_T0=0.6;
      e_n=2.5;
      e_T0_eta=0.05;
      e_n_eta=-0.05;
      eta_p2=1.5;
   }
   if(eta_p2>0.0) {
      fEtaGen=TMath::Power(rnd->Uniform(),eta_p2)*etaMax;
      if(rnd->Uniform()>=0.5) fEtaGen= -fEtaGen;
   } else {
      fEtaGen=rnd->Uniform(-etaMax,etaMax);
   }
   Double_t n=e_n   + e_n_eta*fEtaGen;
   Double_t T0=e_T0 + e_T0_eta*fEtaGen;
   fPtGen=(TMath::Power(rnd->Uniform(),1./(1.-n))-1.)*T0;
   /*   static int print=100;
      if(print) {
         cout<<fEtaGen
            <<" "<<fPtGen
             <<"\n";
         print--;
         } */
}

void ToyEvent::GenerateBgrKinematics(TRandom *rnd,Bool_t isData) {
   fPtGen=0.0;
   fEtaGen=0.0;
   fPtRec=rnd->Exp(isData ? 2.5 : 2.5);
   fEtaRec=rnd->Uniform(-3.,3.);
}

void ToyEvent::GenerateReco(TRandom *rnd) {
   if(fIsSignal) {
      Double_t expEta=TMath::Exp(fEtaGen);
      Double_t eGen=fPtGen*(expEta+1./expEta);
      Double_t sigmaE=0.1*TMath::Sqrt(eGen)+(0.01+0.002*TMath::Abs(fEtaGen))
         *eGen;
      Double_t eRec;
      do {
         eRec=rnd->Gaus(eGen,sigmaE);
      } while(eRec<=0.0);
      Double_t sigmaEta=0.1+0.02*fEtaGen;
      fEtaRec=rnd->Gaus(fEtaGen,sigmaEta);
      fPtRec=eRec/(expEta+1./expEta);
      do {
         Double_t tauDiscr=0.08-0.04/(1.+fPtRec/10.0);
         Double_t sigmaDiscr=0.01;
         fDiscriminator=1.0-rnd->Exp(tauDiscr)+rnd->Gaus(0.,sigmaDiscr);
      } while((fDiscriminator<=0.)||(fDiscriminator>=1.));
      /* static int print=100;
         if(print) {
         cout<<fEtaGen<<" "<<fPtGen
             <<" -> "<<fEtaRec<<" "<<fPtRec
             <<"\n";
         print--;
         } */
   } else {
      do {
         Double_t tauDiscr=0.15-0.05/(1.+fPtRec/5.0)+0.1*fEtaRec;
         Double_t sigmaDiscr=0.02+0.01*fEtaRec;
         fDiscriminator=rnd->Exp(tauDiscr)+rnd->Gaus(0.,sigmaDiscr);
      } while((fDiscriminator<=0.)||(fDiscriminator>=1.));
   }
   fIsTriggered=(rnd->Uniform()<1./(TMath::Exp(-fPtRec+3.5)+1.));
}
