// @(#)root/minuit2:$Name:  $:$Id: TFitterFumili.cxx,v 1.6 2006/07/03 15:48:06 moneta Exp $
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


 
void TFitterFumili::CreateMinimizer(EMinimizerType ) { 
   // Create the minimizer engine and register the plugin in ROOT
   // what ever we specify only Fumili is created 
   if (PrintLevel() >=1 ) 
      std::cout<<"TFitterFumili: Minimize using new Fumili algorithm "<<std::endl;
   
   const ModularFunctionMinimizer * minimizer = GetMinimizer();
   if (!minimizer) delete minimizer;
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
