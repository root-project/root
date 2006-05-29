// @(#)root/minuit2:$Name:  $:$Id: TFitterFumili.cxx,v 1.4 2006/04/26 10:40:09 moneta Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include "TROOT.h"
#include "TFitterFumili.h"
#include "TMath.h"
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
    SetName("Fumili2");
    gFumili2 = this;
    gROOT->GetListOfSpecials()->Add(gFumili2);
}


// needed this additional contructor ? 
TFitterFumili::TFitterFumili(Int_t /* maxpar */) {
    SetName("Fumili2");
    gFumili2 = this;
    gROOT->GetListOfSpecials()->Add(gFumili2);
}


// create the minimizer engine and register the plugin in ROOT
// what ever we specify only Fumili is created  
void TFitterFumili::CreateMinimizer(EMinimizerType ) { 
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
  // do chisquare calculations in case of likelihood fits 
  const TFumiliBinLikelihoodFCN * fcn = dynamic_cast<const TFumiliBinLikelihoodFCN *> ( GetMinuitFCN() ); 
  std::vector<double> p(npar);
  for (int i = 0; i < npar; ++i) 
    p[i] = params[i];
  return fcn->Chi2(p);
}


void TFitterFumili::CreateChi2FCN() { 
  SetMinuitFCN(new TFumiliChi2FCN( *this,GetStrategy()) );
}

void TFitterFumili::CreateChi2ExtendedFCN() { 
  //for Fumili use normal method 
  SetMinuitFCN(new TFumiliChi2FCN(*this, GetStrategy()));
}

void TFitterFumili::CreateBinLikelihoodFCN() { 
  SetMinuitFCN( new TFumiliBinLikelihoodFCN( *this, GetStrategy()) );
}
