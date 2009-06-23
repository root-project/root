// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//___________________________________________________
/*
BEGIN_HTML
<p>
A factory for building PDFs and data for a number counting combination.  
The factory produces a PDF for N channels with uncorrelated background 
uncertainty.  Correlations can be added by extending this PDF with additional terms.
The factory relates the signal in each channel to a master signal strength times the 
expected signal in each channel.  Thus, the final test is performed on the master signal strength.
This yields a more powerful test than letting signal in each channel be independent.
</p>
<p>
The problem has been studied in these references:
<ul>
 <li>   http://arxiv.org/abs/physics/0511028</li>
 <li>   http://arxiv.org/abs/physics/0702156</li>
 <li>   http://cdsweb.cern.ch/record/1099969?ln=en</li>
</ul>
</p>

<p>
One can incorporate uncertainty on the expected signal by adding additional terms.
For the future, perhaps this factory should be extended to include the efficiency terms automatically.
</p>
END_HTML
*/

#ifndef RooStats_NumberCountingPdfFactory
#include "RooStats/NumberCountingPdfFactory.h"
#endif

#ifndef RooStats_RooStatsUtils
#include "RooStats/RooStatsUtils.h"
#endif

#include "RooRealVar.h"
#include "RooAddition.h"
#include "RooProduct.h"
#include "RooDataSet.h"
#include "RooProdPdf.h"
#include "RooFitResult.h"
#include "RooPoisson.h"
#include "RooGlobalFunc.h"
#include "RooCmdArg.h"
#include "RooWorkspace.h"
#include "RooMsgService.h"
#include "TTree.h"
#include <sstream>



ClassImp(RooStats::NumberCountingPdfFactory) ;


using namespace RooStats;
using namespace RooFit;


//_______________________________________________________
NumberCountingPdfFactory::NumberCountingPdfFactory() {
   // constructor

}

//_______________________________________________________
NumberCountingPdfFactory::~NumberCountingPdfFactory(){
   // destructor
}

//_______________________________________________________
void NumberCountingPdfFactory::AddModel(Double_t* sig, 
					Int_t nbins, 
					RooWorkspace* ws, 
					const char* pdfName, const char* muName) {
  

// This method produces a PDF for N channels with uncorrelated background 
// uncertainty. It relates the signal in each channel to a master signal strength times the 
// expected signal in each channel.
//
// For the future, perhaps this method should be extended to include the efficiency terms automatically.

   using namespace RooFit;
   using std::vector;

   TList likelihoodFactors;

   //  Double_t MaxSigma = 8; // Needed to set ranges for varaibles.

   RooRealVar*   masterSignal = 
      new RooRealVar(muName,"masterSignal",1., 0., 3.);


   // loop over individual channels
   for(Int_t i=0; i<nbins; ++i){
      // need to name the variables dynamically, so please forgive the string manipulation and focus on values & ranges.
      std::stringstream str;
      str<<"_"<<i;
      RooRealVar*   expectedSignal = 
         new RooRealVar(("expected_s"+str.str()).c_str(),("expected_s"+str.str()).c_str(),sig[i], 0., 2*sig[i]);
      expectedSignal->setConstant(kTRUE);

      RooProduct*   s = 
         new RooProduct(("s"+str.str()).c_str(),("s"+str.str()).c_str(), RooArgSet(*masterSignal, *expectedSignal)); 

      RooRealVar*   b = 
         new RooRealVar(("b"+str.str()).c_str(),("b"+str.str()).c_str(), .5,  0.,1.);
      RooRealVar*  tau = 
         new RooRealVar(("tau"+str.str()).c_str(),("tau"+str.str()).c_str(), .5, 0., 1.); 
      tau->setConstant(kTRUE);

      RooAddition*  splusb = 
         new RooAddition(("splusb"+str.str()).c_str(),("s"+str.str()+"+"+"b"+str.str()).c_str(),   
                         RooArgSet(*s,*b)); 
      RooProduct*   bTau = 
         new RooProduct(("bTau"+str.str()).c_str(),("b*tau"+str.str()).c_str(),   RooArgSet(*b, *tau)); 
      RooRealVar*   x = 
         new RooRealVar(("x"+str.str()).c_str(),("x"+str.str()).c_str(),  0.5 , 0., 1.);
      RooRealVar*   y = 
         new RooRealVar(("y"+str.str()).c_str(),("y"+str.str()).c_str(),  0.5,  0., 1.);


      RooPoisson* sigRegion = 
         new RooPoisson(("sigRegion"+str.str()).c_str(),("sigRegion"+str.str()).c_str(), *x,*splusb);
      RooPoisson* sideband = 
         new RooPoisson(("sideband"+str.str()).c_str(),("sideband"+str.str()).c_str(), *y,*bTau);

      likelihoodFactors.Add(sigRegion);
      likelihoodFactors.Add(sideband);
    
   }

   RooArgSet likelihoodFactorSet(likelihoodFactors);
   RooProdPdf joint(pdfName,"joint", likelihoodFactorSet );
   //  joint.printCompactTree();

   // add this PDF to workspace.  
   // Need to do import into workspace now to get all the structure imported as well.
   // Just returning the WS will loose the rest of the structure b/c it will go out of scope
   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
   ws->import(joint);
   RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
}

//_______________________________________________________
void NumberCountingPdfFactory::AddExpData(Double_t* sig, 
					  Double_t* back, 
					  Double_t* back_syst, 
					  Int_t nbins, 
					  RooWorkspace* ws, const char* dsName) {

   // Arguements are an array of expected signal, expected background, and relative 
   // background uncertainty (eg. 0.1 for 10% uncertainty), and the number of channels.

   using std::vector;
   Double_t* mainMeas = new Double_t[nbins];

   // loop over channels
   for(Int_t i=0; i<nbins; ++i){
      mainMeas[i] = sig[i] + back[i];
   }
   return AddData(mainMeas, back, back_syst, nbins, ws, dsName);
}

//_______________________________________________________
void NumberCountingPdfFactory::AddExpDataWithSideband(Double_t* sigExp, 
                                                      Double_t* backExp, 
                                                      Double_t* tau, 
                                                      Int_t nbins, 
                                                      RooWorkspace* ws, const char* dsName) {

   // Arguements are an array of expected signal, expected background, and relative 
   // ratio of background expected in the sideband to that expected in signal region, and the number of channels.

   Double_t* mainMeas = new Double_t[nbins];
   Double_t* sideband = new Double_t[nbins];
   for(Int_t i=0; i<nbins; ++i){
      mainMeas[i] = sigExp[i] + backExp[i];
      sideband[i] = backExp[i]*tau[i];
   }
   return AddDataWithSideband(mainMeas, sideband, tau, nbins, ws, dsName);

}

//_______________________________________________________
RooRealVar* NumberCountingPdfFactory::SafeObservableCreation(RooWorkspace* ws, const char* varName, Double_t value) {
   // need to be careful here that the range of observable in the dataset is consistent with the one in the workspace
   // don't rescale unless necessary.  If it is necessary, then rescale by x10 or a defined maximum.

   return SafeObservableCreation(ws, varName, value, 10.*value);

}

//_______________________________________________________
RooRealVar* NumberCountingPdfFactory::SafeObservableCreation(RooWorkspace* ws, const char* varName, 
							     Double_t value, Double_t maximum) {
   // need to be careful here that the range of observable in the dataset is consistent with the one in the workspace
   // don't rescale unless necessary.  If it is necessary, then rescale by x10 or a defined maximum.

   RooRealVar*   x = ws->var( varName );
   if( !x )
      x = new RooRealVar(varName, varName, value, 0, maximum );
   if( x->getMax() < value )
      x->setMax( max(x->getMax(), 10*value ) );
   x->setVal( value );

   return x;
}


//_______________________________________________________
void NumberCountingPdfFactory::AddData(Double_t* mainMeas, 
                                       Double_t* back, 
                                       Double_t* back_syst, 
                                       Int_t nbins, 
                                       RooWorkspace* ws, const char* dsName) {

   // Arguments are an array of results from a main measurement, a measured background,
   //  and relative background uncertainty (eg. 0.1 for 10% uncertainty), and the number of channels.

   using namespace RooFit;
   using std::vector;

   Double_t MaxSigma = 8; // Needed to set ranges for varaibles.

   TList observablesCollection;

   TTree* tree = new TTree();
   Double_t* xForTree = new Double_t[nbins];
   Double_t* yForTree = new Double_t[nbins];

   // loop over channels
   for(Int_t i=0; i<nbins; ++i){
      std::stringstream str;
      str<<"_"<<i;

      Double_t _tau = 1./back[i]/back_syst[i]/back_syst[i];
      RooRealVar*  tau = SafeObservableCreation(ws,  ("tau"+str.str()).c_str(), _tau );

      oocoutW(ws,ObjectHandling) << "NumberCountingPdfFactory: changed value of " << tau->GetName() << " to " << tau->getVal() << 
         " to be consistent with background and its uncertainty. " <<
         " Also stored these values of tau into workspace with name . " << (string(tau->GetName())+string(dsName)).c_str() <<
         " if you test with a different dataset, you should adjust tau appropriately.\n"<< endl;
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      ws->import(*((RooRealVar*) tau->clone( (string(tau->GetName())+string(dsName)).c_str() ) ) );
      RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;

      // need to be careful
      RooRealVar* x = SafeObservableCreation(ws, ("x"+str.str()).c_str(), mainMeas[i]);
    
      // need to be careful
      RooRealVar*   y = SafeObservableCreation(ws, ("y"+str.str()).c_str(), back[i]*_tau );

      observablesCollection.Add(x);
      observablesCollection.Add(y);
    
      xForTree[i] = mainMeas[i];
      yForTree[i] = back[i]*_tau;

      tree->Branch(("x"+str.str()).c_str(), xForTree+i ,("x"+str.str()+"/D").c_str());
      tree->Branch(("y"+str.str()).c_str(), yForTree+i ,("y"+str.str()+"/D").c_str());

      ws->var(("b"+str.str()).c_str())->setMax( 1.2*back[i]+MaxSigma*(sqrt(back[i])+back[i]*back_syst[i]) );
      ws->var(("b"+str.str()).c_str())->setVal( back[i] );

   }
   tree->Fill();
   //  tree->Print();
   //  tree->Scan();

   RooArgList* observableList = new RooArgList(observablesCollection);

   //  observableSet->Print();
   //  observableList->Print();

   RooDataSet* data = new RooDataSet(dsName,"Number Counting Data", tree, *observableList); // one experiment
   //  data->Scan();


   // import hypothetical data
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL) ;
   ws->import(*data);
   RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;

}

//_______________________________________________________
void NumberCountingPdfFactory::AddDataWithSideband(Double_t* mainMeas, 
                                                   Double_t* sideband, 
                                                   Double_t* tauForTree, 
                                                   Int_t nbins, 
                                                   RooWorkspace* ws, const char* dsName) {

   // Arguements are an array of expected signal, expected background, and relative 
   // background uncertainty (eg. 0.1 for 10% uncertainty), and the number of channels.

   using namespace RooFit;
   using std::vector;

   Double_t MaxSigma = 8; // Needed to set ranges for varaibles.

   TList observablesCollection;

   TTree* tree = new TTree();
   Double_t* xForTree = new Double_t[nbins];
   Double_t* yForTree = new Double_t[nbins];

   // loop over channels
   for(Int_t i=0; i<nbins; ++i){
      std::stringstream str;
      str<<"_"<<i;

      Double_t _tau = tauForTree[i];
      Double_t back_syst = 1./sqrt(sideband[i]);
      Double_t back = (sideband[i]/_tau);


      RooRealVar*  tau = SafeObservableCreation(ws,  ("tau"+str.str()).c_str(), _tau );

      oocoutW(ws,ObjectHandling) << "NumberCountingPdfFactory: changed value of " << tau->GetName() << " to " << tau->getVal() << 
         " to be consistent with background and its uncertainty. " <<
         " Also stored these values of tau into workspace with name . " << (string(tau->GetName())+string(dsName)).c_str() <<
         " if you test with a different dataset, you should adjust tau appropriately.\n"<< endl;
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
      ws->import(*((RooRealVar*) tau->clone( (string(tau->GetName())+string(dsName)).c_str() ) ) );
      RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;

      // need to be careful
      RooRealVar* x = SafeObservableCreation(ws, ("x"+str.str()).c_str(), mainMeas[i]);
    
      // need to be careful
      RooRealVar*   y = SafeObservableCreation(ws, ("y"+str.str()).c_str(), sideband[i] );


      observablesCollection.Add(x);
      observablesCollection.Add(y);
    
      xForTree[i] = mainMeas[i];
      yForTree[i] = sideband[i];

      tree->Branch(("x"+str.str()).c_str(), xForTree+i ,("x"+str.str()+"/D").c_str());
      tree->Branch(("y"+str.str()).c_str(), yForTree+i ,("y"+str.str()+"/D").c_str());

      ws->var(("b"+str.str()).c_str())->setMax(  1.2*back+MaxSigma*(sqrt(back)+back*back_syst) );
      ws->var(("b"+str.str()).c_str())->setVal( back );

   }
   tree->Fill();
   //  tree->Print();
   //  tree->Scan();

   RooArgList* observableList = new RooArgList(observablesCollection);

   //  observableSet->Print();
   //  observableList->Print();

   RooDataSet* data = new RooDataSet(dsName,"Number Counting Data", tree, *observableList); // one experiment
   //  data->Scan();


   // import hypothetical data
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL) ;
   ws->import(*data);
   RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;

}



