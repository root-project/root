// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////
// NumberCountingUtils
//
// Encapsulates common number counting utilities
/////////////////////////////////////////
  ///////////////////////////////////
  // Standalone Functions.
  // Naming conventions:
  //  Exp = Expected
  //  Obs = Observed
  //  P   = p-value
  //  Z   = Z-value or significance in Sigma (one-sided convention)
  //////////////////////////////////

#include "NumberCountingUtils.h"


// Without this macro the THtml doc for TMath can not be generated
#if !defined(R__ALPHA) && !defined(R__SOLARIS) && !defined(R__ACC) && !defined(R__FBSD)
NamespaceImp(NumberCountingUtils)
#endif

Double_t NumberCountingUtils::BinomialExpP(Double_t signalExp, Double_t backgroundExp, Double_t relativeBkgUncert){
  // Expected P-value for s=0 in a ratio of Poisson means.  
  // Here the background and its uncertainty are provided directly and 
  // assumed to be from the double Poisson counting setup described in the 
  // BinomialWithTau functions.  
  // Normally one would know tau directly, but here it is determiend from
  // the background uncertainty.  This is not strictly correct, but a useful 
  // approximation.
  //
  //
  // This is based on code and comments from Bob Cousins 
  //  based on the following papers:
  //
  // Statistical Challenges for Searches for New Physics at the LHC
  // Authors: Kyle Cranmer
  // http://arxiv.org/abs/physics/0511028
  //
  //  Measures of Significance in HEP and Astrophysics
  //  Authors: J. T. Linnemann
  //  http://arxiv.org/abs/physics/0312059
  //
  // In short, this is the exact frequentist solution to the problem of
  // a main measurement x distributed as a Poisson around s+b and a sideband or 
  // auxiliary measurement y distributed as a Poisson around \taub.  Eg. 
  // L(x,y|s,b,\tau) = Pois(x|s+b)Pois(y|\tau b)


  //SIDE BAND EXAMPLE
  //See Eqn. (19) of Cranmer and pp. 36-37 of Linnemann.
  //150 total events in signalExp region, 100 in sideband of equal size
  Double_t mainInf = signalExp+backgroundExp;  //Given
  Double_t tau = 1./backgroundExp/(relativeBkgUncert*relativeBkgUncert);
  Double_t auxiliaryInf = backgroundExp*tau;  //Given
  
  Double_t P_Bi = TMath::BetaIncomplete(1./(1.+tau),mainInf,auxiliaryInf+1);
  return P_Bi;
  
/*
Now, if instead the mean background level b in the signal region is
specified, along with Gaussian rms sigb, then one can fake a Poisson
sideband (see Linnemann, p. 35, converted to Cranmer's notation) by
letting tau = b/(sigb*sigb) and y = b*tau.  Thus, for example, if one
has x=150 and b = 100 +/- 10, one then derives tau and y.  Then one
has the same two lines of ROOT calling BetaIncomplete and ErfInverse.
Since I chose these numbers to revert to the previous example, we get
the same answer:
*/
/*
//GAUSSIAN FAKED AS POISSON EXAMPLE
x = 150.    //Given
b = 100.    //Given
sigb = 10.  //Given
tau = b/(sigb*sigb)
y = tau*b   
Z_Bi = TMath::BetaIncomplete(1./(1.+tau),x,y+1)    
S = sqrt(2)*TMath::ErfInverse(1 - 2*Z_Bi)     

*/

}


Double_t NumberCountingUtils::BinomialWithTauExpP(Double_t signalExp, Double_t backgroundExp, Double_t tau){
  // Expected P-value for s=0 in a ratio of Poisson means.  
  // Based on two expectations, a main measurement that might have signal
  // and an auxiliarly measurement for the background that is signal free.
  // The expected background in the auxiliary measurement is a factor
  // tau larger than in the main measurement.

  Double_t mainInf = signalExp+backgroundExp;  //Given
  Double_t auxiliaryInf = backgroundExp*tau;  //Given
  
  Double_t P_Bi = TMath::BetaIncomplete(1./(1.+tau),mainInf,auxiliaryInf+1);
  
  return P_Bi;
  
}

Double_t NumberCountingUtils::BinomialObsP(Double_t mainObs, Double_t backgroundObs, Double_t relativeBkgUncert){
  // P-value for s=0 in a ratio of Poisson means.  
  // Here the background and its uncertainty are provided directly and 
  // assumed to be from the double Poisson counting setup.  
  // Normally one would know tau directly, but here it is determiend from
  // the background uncertainty.  This is not strictly correct, but a useful 
  // approximation.
  
  Double_t tau = 1./backgroundObs/(relativeBkgUncert*relativeBkgUncert);
  Double_t auxiliaryInf = backgroundObs*tau;  //Given
  
    
  //SIDE BAND EXAMPLE
  //See Eqn. (19) of Cranmer and pp. 36-37 of Linnemann.
  Double_t P_Bi = TMath::BetaIncomplete(1./(1.+tau),mainObs,auxiliaryInf+1);
  
  return P_Bi;

}


Double_t NumberCountingUtils::BinomialWithTauObsP(Double_t mainObs, Double_t auxiliaryObs, Double_t tau){
  // P-value for s=0 in a ratio of Poisson means.  
  // Based on two observations, a main measurement that might have signal
  // and an auxiliarly measurement for the background that is signal free.
  // The expected background in the auxiliary measurement is a factor
  // tau larger than in the main measurement.

  //SIDE BAND EXAMPLE
  //See Eqn. (19) of Cranmer and pp. 36-37 of Linnemann.
  Double_t P_Bi = TMath::BetaIncomplete(1./(1.+tau),mainObs,auxiliaryObs+1);
  
  return P_Bi;
  
}

Double_t NumberCountingUtils::BinomialExpZ(Double_t signalExp, Double_t backgroundExp, Double_t relativeBkgUncert) {    
  // See BinomialExpP
  return Statistics::PValueToSignificance( BinomialExpP(signalExp,backgroundExp,relativeBkgUncert) ) ;
  }

Double_t NumberCountingUtils::BinomialWithTauExpZ(Double_t signalExp, Double_t backgroundExp, Double_t tau){
  // See BinomialWithTauExpP
  return Statistics::PValueToSignificance( BinomialWithTauExpP(signalExp,backgroundExp,tau) ) ;
}


Double_t NumberCountingUtils::BinomialObsZ(Double_t mainObs, Double_t backgroundObs, Double_t relativeBkgUncert){
  // See BinomialObsZ
  return Statistics::PValueToSignificance( BinomialObsP(mainObs,backgroundObs,relativeBkgUncert) ) ;
}

Double_t NumberCountingUtils::BinomialWithTauObsZ(Double_t mainObs, Double_t auxiliaryObs, Double_t tau){
  // See BinomialWithTauObsZ
  return Statistics::PValueToSignificance( BinomialWithTauObsZ(mainObs,auxiliaryObs,tau) ) ;  
}

/////////////////////////////////////////////////////////////
//
//  RooFit based Functions
//
/////////////////////////////////////////////////////////////
#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooAddition.h"
#include "RooProduct.h"
#include "RooDataSet.h"
#include "RooProdPdf.h"
#include "RooPlot.h"
#include "TMath.h"
#include "TCanvas.h"
#include "RooFitResult.h"
#include "RooNLLVar.h"
#include "RooPoisson.h"
#include "RooGaussian.h"
#include "RooMinuit.h"
#include "RooGlobalFunc.h"
#include "RooCmdArg.h"
#include "TH2F.h"
#include "TTree.h"
#include <sstream>


Double_t NumberCountingUtils::ProfileCombinationExpZ(Double_t* sig, 
						     Double_t* back, 
						     Double_t* back_syst, 
						     Int_t nbins){

  // A number counting combination for N channels with uncorrelated background 
  //  uncertainty.  
  // Background uncertainty taken into account via the profile likelihood ratio.
  // Arguements are an array of expected signal, expected background, and relative 
  // background uncertainty (eg. 0.1 for 10% uncertainty), and the number of channels.

  using namespace RooFit;
  using std::vector;

  vector<RooRealVar*> backVec, tauVec, xVec, yVec;
  vector<RooProduct*> sigVec;
  vector<RooFormulaVar*> splusbVec;
  vector<RooPoisson*> sigRegions, sidebands;
  TList likelihoodFactors;
  TList observablesCollection;

  TTree* tree = new TTree();
  Double_t* xForTree = new Double_t[nbins];
  Double_t* yForTree = new Double_t[nbins];

  Double_t MaxSigma = 8; // Needed to set ranges for varaibles.

  RooRealVar*   masterSignal = 
    new RooRealVar("masterSignal","masterSignal",1., 0., 3.);
  for(Int_t i=0; i<nbins; ++i){
    std::stringstream str;
    str<<"_"<<i;
    RooRealVar*   expectedSignal = 
      new RooRealVar(("expected_s"+str.str()).c_str(),("expected_s"+str.str()).c_str(),sig[i], 0., 2*sig[i]);
    expectedSignal->setConstant(kTRUE);

    RooProduct*   s = 
      new RooProduct(("s"+str.str()).c_str(),("s"+str.str()).c_str(), RooArgSet(*masterSignal, *expectedSignal)); 

    RooRealVar*   b = 
      new RooRealVar(("b"+str.str()).c_str(),("b"+str.str()).c_str(),back[i],  0., 1.2*back[i]+MaxSigma*(sqrt(back[i])+back[i]*back_syst[i]));
    b->Print();
    Double_t _tau = 1./back[i]/back_syst[i]/back_syst[i];
    RooRealVar*  tau = 
      new RooRealVar(("tau"+str.str()).c_str(),("tau"+str.str()).c_str(),_tau,0,2*_tau); 
    tau->setConstant(kTRUE);

    RooAddition*  splusb = 
      new RooAddition(("splusb"+str.str()).c_str(),("s"+str.str()+"+"+"b"+str.str()).c_str(),   
		      RooArgSet(*s,*b)); 
    RooProduct*   bTau = 
      new RooProduct(("bTau"+str.str()).c_str(),("b*tau"+str.str()).c_str(),   RooArgSet(*b, *tau)); 
    RooRealVar*   x = 
      new RooRealVar(("x"+str.str()).c_str(),("x"+str.str()).c_str(),  sig[i]+back[i], 0., 1.2*sig[i]+back[i]+MaxSigma*sqrt(sig[i]+back[i]));
    RooRealVar*   y = 
      new RooRealVar(("y"+str.str()).c_str(),("y"+str.str()).c_str(),  back[i]*_tau,  0., 1.2*back[i]*_tau+MaxSigma*sqrt(back[i]*_tau));


    RooPoisson* sigRegion = 
      new RooPoisson(("sigRegion"+str.str()).c_str(),("sigRegion"+str.str()).c_str(), *x,*splusb);
    RooPoisson* sideband = 
      new RooPoisson(("sideband"+str.str()).c_str(),("sideband"+str.str()).c_str(), *y,*bTau);

    sigVec.push_back(s);
    backVec.push_back(b);
    tauVec.push_back(tau);
    xVec.push_back(x);
    yVec.push_back(y);
    sigRegions.push_back(sigRegion);
    sidebands.push_back(sideband);

    likelihoodFactors.Add(sigRegion);
    likelihoodFactors.Add(sideband);
    observablesCollection.Add(x);
    observablesCollection.Add(y);
    
    // print to see range on variables
    //    x->Print();
    //    y->Print();
    //    b->Print();


    xForTree[i] = sig[i]+back[i];
    yForTree[i] = back[i]*_tau;
    tree->Branch(("x"+str.str()).c_str(), xForTree+i ,("x"+str.str()+"/D").c_str());
    tree->Branch(("y"+str.str()).c_str(), yForTree+i ,("y"+str.str()+"/D").c_str());
  }
  tree->Fill();
  //  tree->Print();
  //  tree->Scan();

  RooArgSet likelihoodFactorSet(likelihoodFactors);
  RooProdPdf joint("joint","joint", likelihoodFactorSet );
  //  likelihoodFactorSet.Print();

  //  cout << "\n print model" << endl;
  //  joint.Print();
  //  joint.printCompactTree();

  //  RooArgSet* observableSet = new RooArgSet(observablesCollection);
  RooArgList* observableList = new RooArgList(observablesCollection);

  //  observableSet->Print();
  //  observableList->Print();

  //  cout << "Make hypothetical dataset:" << endl;
  RooDataSet* toyMC = new RooDataSet("data","data", tree, *observableList); // one experiment
  toyMC->Scan();

  //  cout << "about to do fit \n\n" << endl;
  RooFitResult* fit = joint.fitTo(*toyMC,Extended(kFALSE),Strategy(0),Hesse(kFALSE),Save(kTRUE),PrintLevel(-1));

  //RooFitResult* fit = joint.fitTo(*toyMC,"sr");
  //  fit->Print();

  //  joint.Print("v");

  ////////////////////////////////////////
  /// Calculate significance
  //////////////////////////////
  //  cout << "\nFit to signal plus background:" << endl;
  masterSignal->Print();
  for(Int_t i=0; i<nbins; ++i) backVec.at(i)->Print();
  fit->Print();
  Double_t NLLatMLE= fit->minNll();



  //  cout << "\nFit to background only:" << endl;
  masterSignal->setVal(0);
  masterSignal->setConstant();
  RooFitResult* fit2 = joint.fitTo(*toyMC,Extended(kFALSE),Hesse(kFALSE),Strategy(0), Minos(kFALSE), Save(kTRUE),PrintLevel(-1));

  masterSignal->Print();
  for(Int_t i=0; i<nbins; ++i) backVec.at(i)->Print();
  Double_t NLLatCondMLE= fit2->minNll();
  fit2->Print();

  return sqrt( 2*(NLLatCondMLE-NLLatMLE)); 

}
