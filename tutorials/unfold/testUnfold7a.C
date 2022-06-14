/// \file
/// \ingroup tutorial_unfold
/// \notebook
/// Test program for the classes TUnfoldDensity and TUnfoldBinning.
///
/// A toy test of the TUnfold package
///
/// This example is documented in conference proceedings:
///
///   arXiv:1611.01927
///   12th Conference on Quark Confinement and the Hadron Spectrum (Confinement XII)
///
/// This is an example of unfolding a two-dimensional distribution
/// also using an auxiliary measurement to constrain some background
///
/// The example comprises several macros
///  - testUnfold7a.C   create root files with TTree objects for
///                      signal, background and data
///            - write files  testUnfold7_signal.root
///                           testUnfold7_background.root
///                           testUnfold7_data.root
///
///  - testUnfold7b.C   loop over trees and fill histograms based on the
///                      TUnfoldBinning objects
///            - read  testUnfold7binning.xml
///                    testUnfold7_signal.root
///                    testUnfold7_background.root
///                    testUnfold7_data.root
///
///            - write testUnfold7_histograms.root
///
///  - testUnfold7c.C   run the unfolding
///            - read  testUnfold7_histograms.root
///            - write testUnfold7_result.root
///                    testUnfold7_result.ps
///
/// \macro_output
/// \macro_code
///
///  **Version 17.6, in parallel to changes in TUnfold**
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
#include <TLorentzVector.h>

#define MASS1 0.511E-3

using namespace std;

TRandom *g_rnd=0;

class ToyEvent7 {
public:
   void GenerateDataEvent(TRandom *rnd);
   void GenerateSignalEvent(TRandom *rnd);
   void GenerateBgrEvent(TRandom *rnd);
   // reconstructed quantities
   inline Double_t GetMRec(int i) const { return fMRec[i]; }
   inline Double_t GetPtRec(int i) const { return fPtRec[i]; }
   inline Double_t GetEtaRec(int i) const { return fEtaRec[i]; }
   inline Double_t GetDiscriminator(void) const {return fDiscriminator; }
   inline Double_t GetPhiRec(int i) const { return fPhiRec[i]; }
   inline Bool_t IsTriggered(void) const { return fIsTriggered; }

   // generator level quantities
   inline Double_t GetMGen(int i) const {
      if(IsSignal()) return fMGen[i];
      else return -1.0;
   }
   inline Double_t GetPtGen(int i) const {
      if(IsSignal()) return fPtGen[i];
      else return -1.0;
   }
   inline Double_t GetEtaGen(int i) const {
       if(IsSignal()) return fEtaGen[i];
       else return 999.0;
   }
   inline Double_t GetPhiGen(int i) const {
       if(IsSignal()) return fPhiGen[i];
       else return 999.0;
   }
   inline Bool_t IsSignal(void) const { return fIsSignal; }
protected:

   void GenerateSignalKinematics(TRandom *rnd,Bool_t isData);
   void GenerateBgrKinematics(TRandom *rnd,Bool_t isData);
   void GenerateReco(TRandom *rnd);

   // reconstructed quantities
   Double_t fMRec[3];
   Double_t fPtRec[3];
   Double_t fEtaRec[3];
   Double_t fPhiRec[3];
   Double_t fDiscriminator;
   Bool_t fIsTriggered;
   // generated quantities
   Double_t fMGen[3];
   Double_t fPtGen[3];
   Double_t fEtaGen[3];
   Double_t fPhiGen[3];
   Bool_t fIsSignal;
public:
   static Double_t kDataSignalFraction;
   static Double_t kMCSignalFraction;

};

void testUnfold7a()
{
  // random generator
  g_rnd=new TRandom3(4711);

  // data and MC number of events
  Double_t muData0=5000.;
  // luminosity error
  Double_t muData=muData0*g_rnd->Gaus(1.0,0.03);
  // stat error
  Int_t neventData      = g_rnd->Poisson( muData);

  // generated number of MC events
  Int_t neventSigmc     = 250000;
  Int_t neventBgrmc     = 100000;

  Float_t etaRec[3],ptRec[3],phiRec[3],mRec[3],discr;
  Float_t etaGen[3],ptGen[3],phiGen[3],mGen[3];
  Float_t weight;
  Int_t istriggered,issignal;

  //==================================================================
  // Step 1: generate data TTree

  TFile *dataFile=new TFile("testUnfold7_data.root","recreate");
  TTree *dataTree=new TTree("data","event");

  dataTree->Branch("etarec",etaRec,"etarec[3]/F");
  dataTree->Branch("ptrec",ptRec,"ptrec[3]/F");
  dataTree->Branch("phirec",phiRec,"phirec[3]/F");
  dataTree->Branch("mrec",mRec,"mrec[3]/F");
  dataTree->Branch("discr",&discr,"discr/F");

  // for real data, only the triggered events are available
  dataTree->Branch("istriggered",&istriggered,"istriggered/I");
  // data truth parameters
  dataTree->Branch("etagen",etaGen,"etagen[3]/F");
  dataTree->Branch("ptgen",ptGen,"ptgen[3]/F");
  dataTree->Branch("phigen",phiGen,"phigen[3]/F");
  dataTree->Branch("mgen",mGen,"mgen[3]/F");
  dataTree->Branch("issignal",&issignal,"issignal/I");

  cout<<"fill data tree\n";

  //Int_t nEvent=0,nTriggered=0;
  for(int ievent=0;ievent<neventData;ievent++) {
     ToyEvent7 event;
     event.GenerateDataEvent(g_rnd);
     for(int i=0;i<3;i++) {
        etaRec[i]=event.GetEtaRec(i);
        ptRec[i]=event.GetPtRec(i);
        phiRec[i]=event.GetPhiRec(i);
        mRec[i]=event.GetMRec(i);
        etaGen[i]=event.GetEtaGen(i);
        ptGen[i]=event.GetPtGen(i);
        phiGen[i]=event.GetPhiGen(i);
        mGen[i]=event.GetMGen(i);
     }
     discr=event.GetDiscriminator();
     istriggered=event.IsTriggered() ? 1 : 0;
     issignal=event.IsSignal() ? 1 : 0;

     dataTree->Fill();

     if(!(ievent%100000)) cout<<"   data event "<<ievent<<"\n";

     //if(istriggered) nTriggered++;
     //nEvent++;

  }

  dataTree->Write();
  delete dataTree;
  delete dataFile;

  //==================================================================
  // Step 2: generate signal TTree

  TFile *signalFile=new TFile("testUnfold7_signal.root","recreate");
  TTree *signalTree=new TTree("signal","event");

  signalTree->Branch("etarec",etaRec,"etarec[3]/F");
  signalTree->Branch("ptrec",ptRec,"ptrec[3]/F");
  signalTree->Branch("phirec",ptRec,"phirec[3]/F");
  signalTree->Branch("mrec",mRec,"mrec[3]/F");
  signalTree->Branch("discr",&discr,"discr/F");

  // for real data, only the triggered events are available
  signalTree->Branch("istriggered",&istriggered,"istriggered/I");
  // data truth parameters
  signalTree->Branch("etagen",etaGen,"etagen[3]/F");
  signalTree->Branch("ptgen",ptGen,"ptgen[3]/F");
  signalTree->Branch("phigen",phiGen,"phigen[3]/F");
  signalTree->Branch("weight",&weight,"weight/F");
  signalTree->Branch("mgen",mGen,"mgen[3]/F");

  cout<<"fill signal tree\n";

  weight=ToyEvent7::kMCSignalFraction*muData0/neventSigmc;

  for(int ievent=0;ievent<neventSigmc;ievent++) {
     ToyEvent7 event;
     event.GenerateSignalEvent(g_rnd);

     for(int i=0;i<3;i++) {
        etaRec[i]=event.GetEtaRec(i);
        ptRec[i]=event.GetPtRec(i);
        phiRec[i]=event.GetPhiRec(i);
        mRec[i]=event.GetMRec(i);
        etaGen[i]=event.GetEtaGen(i);
        ptGen[i]=event.GetPtGen(i);
        phiGen[i]=event.GetPhiGen(i);
        mGen[i]=event.GetMGen(i);
     }
     discr=event.GetDiscriminator();
     istriggered=event.IsTriggered() ? 1 : 0;

     if(!(ievent%100000)) cout<<"   signal event "<<ievent<<"\n";

     signalTree->Fill();
  }

  signalTree->Write();
  delete signalTree;
  delete signalFile;

  // ==============================================================
  // Step 3: generate background MC TTree

  TFile *bgrFile=new TFile("testUnfold7_background.root","recreate");
  TTree *bgrTree=new TTree("background","event");

  bgrTree->Branch("etarec",&etaRec,"etarec[3]/F");
  bgrTree->Branch("ptrec",&ptRec,"ptrec[3]/F");
  bgrTree->Branch("phirec",&phiRec,"phirec[3]/F");
  bgrTree->Branch("mrec",&mRec,"mrec[3]/F");
  bgrTree->Branch("discr",&discr,"discr/F");
  bgrTree->Branch("istriggered",&istriggered,"istriggered/I");
  bgrTree->Branch("weight",&weight,"weight/F");

  cout<<"fill background tree\n";

  weight=(1.-ToyEvent7::kMCSignalFraction)*muData0/neventBgrmc;

  for(int ievent=0;ievent<neventBgrmc;ievent++) {
     ToyEvent7 event;
     event.GenerateBgrEvent(g_rnd);
     for(int i=0;i<3;i++) {
        etaRec[i]=event.GetEtaRec(i);
        ptRec[i]=event.GetPtRec(i);
        phiRec[i]=event.GetPhiRec(i);
     }
     discr=event.GetDiscriminator();
     istriggered=event.IsTriggered() ? 1 : 0;

     if(!(ievent%100000)) cout<<"   background event "<<ievent<<"\n";

     bgrTree->Fill();
  }

  bgrTree->Write();
  delete bgrTree;
  delete bgrFile;
}

Double_t ToyEvent7::kDataSignalFraction=0.75;
Double_t ToyEvent7::kMCSignalFraction=0.75;

void ToyEvent7::GenerateDataEvent(TRandom *rnd) {
   fIsSignal=rnd->Uniform()<kDataSignalFraction;
   if(IsSignal()) {
      GenerateSignalKinematics(rnd,kTRUE);
   } else {
      GenerateBgrKinematics(rnd,kTRUE);
   }
   GenerateReco(rnd);
}

void ToyEvent7::GenerateSignalEvent(TRandom *rnd) {
   fIsSignal=1;
   GenerateSignalKinematics(rnd,kFALSE);
   GenerateReco(rnd);
}

void ToyEvent7::GenerateBgrEvent(TRandom *rnd) {
   fIsSignal=0;
   GenerateBgrKinematics(rnd,kFALSE);
   GenerateReco(rnd);
}

void ToyEvent7::GenerateSignalKinematics(TRandom *rnd,Bool_t isData) {

   // fake decay of Z0 to two fermions
   double M0=91.1876;
   double Gamma=2.4952;
   // generated mass
   do {
      fMGen[2]=rnd->BreitWigner(M0,Gamma);
   } while(fMGen[2]<=0.0);

   double N_ETA=3.0;
   double MU_PT=5.;
   double SIGMA_PT=2.0;
   double DECAY_A=0.2;
   if(isData) {
      //N_ETA=2.5;
      MU_PT=6.;
      SIGMA_PT=1.8;
      //DECAY_A=0.5;
   }
   fEtaGen[2]=TMath::Power(rnd->Uniform(0,1.5),N_ETA);
   if(rnd->Uniform(-1.,1.)<0.) fEtaGen[2] *= -1.;
   fPhiGen[2]=rnd->Uniform(-M_PI,M_PI);
   do {
      fPtGen[2]=rnd->Landau(MU_PT,SIGMA_PT);
   } while((fPtGen[2]<=0.0)||(fPtGen[2]>500.));
   //========================== decay
   TLorentzVector sum;
   sum.SetPtEtaPhiM(fPtGen[2],fEtaGen[2],fPhiGen[2],fMGen[2]);
   // boost into lab-frame
   TVector3 boost=sum.BoostVector();
   // decay in rest-frame

   TLorentzVector p[3];
   double m=MASS1;
   double costh;
   do {
      double r=rnd->Uniform(-1.,1.);
      costh=r*(1.+DECAY_A*r*r);
   } while(fabs(costh)>=1.0);
   double phi=rnd->Uniform(-M_PI,M_PI);
   double e=0.5*sum.M();
   double ptot=TMath::Sqrt(e+m)*TMath::Sqrt(e-m);
   double pz=ptot*costh;
   double pt=TMath::Sqrt(ptot+pz)*TMath::Sqrt(ptot-pz);
   double px=pt*cos(phi);
   double py=pt*sin(phi);
   p[0].SetXYZT(px,py,pz,e);
   p[1].SetXYZT(-px,-py,-pz,e);
   for(int i=0;i<2;i++) {
      p[i].Boost(boost);
   }
   p[2]=p[0]+p[1];
   for(int i=0;i<3;i++) {
      fPtGen[i]=p[i].Pt();
      fEtaGen[i]=p[i].Eta();
      fPhiGen[i]=p[i].Phi();
      fMGen[i]=p[i].M();
   }
}

void ToyEvent7::GenerateBgrKinematics(TRandom *rnd,Bool_t isData) {
   for(int i=0;i<3;i++) {
      fPtGen[i]=0.0;
      fEtaGen[i]=0.0;
      fPhiGen[i]=0.0;
   }
   TLorentzVector p[3];
   for(int i=0;i<2;i++) {
      p[i].SetPtEtaPhiM(rnd->Exp(15.0),rnd->Uniform(-3.,3.),
                        rnd->Uniform(-M_PI,M_PI),isData ? MASS1 : MASS1);
   }
   p[2]=p[0]+p[1];
   for(int i=0;i<3;i++) {
      fPtRec[i]=p[i].Pt();
      fEtaRec[i]=p[i].Eta();
      fPhiRec[i]=p[i].Phi();
      fMRec[i]=p[i].M();
   }
}

void ToyEvent7::GenerateReco(TRandom *rnd) {
   if(fIsSignal) {
      TLorentzVector p[3];
      for(int i=0;i<2;i++) {
         Double_t expEta=TMath::Exp(fEtaGen[i]);
         Double_t coshEta=(expEta+1./expEta);
         Double_t eGen=fPtGen[i]*coshEta;
         Double_t sigmaE=
            0.1*TMath::Sqrt(eGen)
            +1.0*coshEta
            +0.01*eGen;
         Double_t eRec;
         do {
            eRec=rnd->Gaus(eGen,sigmaE);
         } while(eRec<=0.0);
         Double_t sigmaEta=0.1+0.02*TMath::Abs(fEtaGen[i]);
         p[i].SetPtEtaPhiM(eRec/(expEta+1./expEta),
                           rnd->Gaus(fEtaGen[i],sigmaEta),
                           remainder(rnd->Gaus(fPhiGen[i],0.03),2.*M_PI),
                           MASS1);
      }
      p[2]=p[0]+p[1];
      for(int i=0;i<3;i++) {
         fPtRec[i]=p[i].Pt();
         fEtaRec[i]=p[i].Eta();
         fPhiRec[i]=p[i].Phi();
         fMRec[i]=p[i].M();
      }
   }
   if(fIsSignal) {
      do {
         Double_t tauDiscr=0.08-0.04/(1.+fPtRec[2]/10.0);
         Double_t sigmaDiscr=0.01;
         fDiscriminator=1.0-rnd->Exp(tauDiscr)+rnd->Gaus(0.,sigmaDiscr);
      } while((fDiscriminator<=0.)||(fDiscriminator>=1.));
   } else {
      do {
         Double_t tauDiscr=0.15-0.05/(1.+fPtRec[2]/5.0)+0.1*fEtaRec[2];
         Double_t sigmaDiscr=0.02+0.01*fEtaRec[2];
         fDiscriminator=rnd->Exp(tauDiscr)+rnd->Gaus(0.,sigmaDiscr);
      } while((fDiscriminator<=0.)||(fDiscriminator>=1.));
   }
   fIsTriggered=false;
   for(int i=0;i<2;i++) {
      if(rnd->Uniform()<0.92/(TMath::Exp(-(fPtRec[i]-15.5)/2.5)+1.)) fIsTriggered=true;
   }
}
