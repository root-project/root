// @(#)root/minuit2:$Name:  $:$Id: TBinLikelihoodFCN.cxx,v 1.2 2005/11/05 15:17:35 moneta Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include "TBinLikelihoodFCN.h"
#include "TChi2FitData.h"
#include "FitterUtil.h"

#include <cmath>
#include <cassert>

#include "TF1.h"
#include "TVirtualFitter.h"

//#define DEBUG 1
#ifdef DEBUG
#include <iostream>
#endif

// constructor _ create FitData class

TBinLikelihoodFCN::TBinLikelihoodFCN( const TVirtualFitter & fitter) : 
  fUp(0.5), fOwner(true)
{ 
  fFunc = dynamic_cast<TF1 *> ( fitter.GetUserFunc() );
  assert(fFunc != 0);
  // to do: use class for likelihood data (errors are not necessary)
  // in likelihood fit need to keep empty bins
  fData = new TChi2FitData(fitter, false); 
#ifdef DEBUG
  std::cout << "Created FitData with size = " << fData->Size() << std::endl;
#endif

  // need to set the size so ROOT can calculate ndf.
  fFunc->SetNumberFitPoints(fData->Size());
}

//  this class manages the fit data class. Delete it at the end

TBinLikelihoodFCN::~TBinLikelihoodFCN() {  
  if (fOwner && fData) { 
#ifdef DEBUG
    std::cout << "deleting the data - size is " << fData->Size() << std::endl; 
#endif
    delete fData; 
  }
}


  // implement chi2 function 
double TBinLikelihoodFCN::operator()(const std::vector<double>& par) const {

  assert(fData != 0); 
  assert(fFunc != 0); 

  // safety measure against negative logs
  static const double epsilon = 1e-300;

  //  std::cout << "number of params " << par.size() << " in TF1 " << fFunc->GetNpar() << "  " << fFunc->GetNumberFreeParameters() << std::endl;
  
  unsigned int n = fData->Size();
  double loglike = 0;
  for (unsigned int i = 0; i < n; ++ i) { 
    const std::vector<double> & x = fData->Coords(i); 
    double y = fData->Value(i);
    //std::cout << x[0] << "  " << y << "  " << 1./invError << " params " << par[0] << std::endl;
    double fval;
    if (fData->UseIntegral()) { 
      const std::vector<double> & x2 = fData->Coords(i+1);
      fval = FitterUtil::EvalIntegral(fFunc,x,x2,par);
    }
    else   
      fval = fFunc->EvalPar( &x.front(), &par.front() ); 

    double logtmp;
    // protections against negative argument to the log 
    // smooth linear extrapolation below pml_A
    if(fval<=epsilon) logtmp = fval/epsilon + std::log(epsilon) - 1; 
    else       logtmp = std::log(fval);

    loglike +=  fval - y*logtmp;  
  }

  return loglike;
}

