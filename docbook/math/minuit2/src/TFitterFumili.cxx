// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include "TROOT.h"
#include "TFitterFumili.h"
#include "TF1.h"
#include "TH1.h"
#include "TGraph.h"

#include "TFumiliFCN.h"
#include "Minuit2/FumiliMinimizer.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnPrint.h"

//______________________________________________________________________________
/**
Implementation of the  TVirtualFitter interface using the FUMILI algorithm present in Minuit2. 
The FUMILI algorithm is a specialized minimization algorithm in the case of a least square or likelihood 
functions. With this technique the Hessian matrix of the objective function is approximated using only first 
derivatives.     
BEGIN_HTML
For more information on this method see page 31 of the <a href="http://seal.cern.ch/documents/minuit/mntutorial.pdf">Tutorial on Function Minimization</a>. 
<p>
The new Fumili (Fumili2) can be set as the default fitter to be used in method like TH1::Fit, by doing 
<pre>
TVirtualFitter::SetDefaultFitter("Fumili2");
</pre>
To be used directly, Fumili2 requires that the user implements the ROOT::Minuit2::FumiliFCNBase interface and passes its pointer via the SetMinuitFCN method.
END_HTML
*/



using namespace ROOT::Minuit2;

//#define DEBUG 1

ClassImp(TFitterFumili);

TFitterFumili* gFumili2 = 0;

TFitterFumili::TFitterFumili() {
   // Constructor. Set Name and global pointer. 
   SetName("Fumili2");
   gFumili2 = this;
   gROOT->GetListOfSpecials()->Add(gFumili2);
}



TFitterFumili::TFitterFumili(Int_t /* maxpar */) {
   // Constructor as default. Needed this for TVirtualFitter interface 
   SetName("Fumili2");
   gFumili2 = this;
   gROOT->GetListOfSpecials()->Add(gFumili2);
}


TFitterFumili::~TFitterFumili() { 
// destructor - deletes the minimizer 

   // delete pointer from TROOT
   gROOT->GetListOfSpecials()->Remove(this);
   if (gFumili2 == this) gFumili2 = 0;
   
}

 
void TFitterFumili::CreateMinimizer(EMinimizerType ) { 
   // Create the minimizer engine and register the plugin in ROOT
   // what ever we specify only Fumili is created 
   if (PrintLevel() >=1 ) 
      std::cout<<"TFitterFumili: Minimize using new Fumili algorithm "<<std::endl;
   
   const ModularFunctionMinimizer * minimizer = GetMinimizer();
   if (minimizer) delete minimizer;
   SetMinimizer( new FumiliMinimizer() );
   
   SetStrategy(1);
   // Fumili cannot deal with tolerance too smalls (10-3 corrsponds to 10-7 in FumiliBuilder)
   SetMinimumTolerance(0.001); 
   
#ifdef DEBUG
   SetPrintLevel(3);
#endif
}


Double_t TFitterFumili::Chisquare(Int_t npar, Double_t *params) const {
   // Do chisquare calculations in case of likelihood fits. 
   const TFumiliBinLikelihoodFCN * fcn = dynamic_cast<const TFumiliBinLikelihoodFCN *> ( GetMinuitFCN() ); 
   std::vector<double> p(npar);
   for (int i = 0; i < npar; ++i) 
      p[i] = params[i];
   return fcn->Chi2(p);
}


void TFitterFumili::CreateChi2FCN() { 
   // Create Chi2FCN Fumili function.
   SetMinuitFCN(new TFumiliChi2FCN( *this,GetStrategy()) );
}

void TFitterFumili::CreateChi2ExtendedFCN() { 
   //ExtendedFCN: for Fumili use normal method. 
   SetMinuitFCN(new TFumiliChi2FCN(*this, GetStrategy()));
}

void TFitterFumili::CreateBinLikelihoodFCN() { 
   // Create bin likelihood FCN for Fumili. 
   SetMinuitFCN( new TFumiliBinLikelihoodFCN( *this, GetStrategy()) );
}
