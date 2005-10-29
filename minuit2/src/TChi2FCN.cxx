// @(#)root/minuit2:$Name:  $:$Id: TChi2FCN.cxx,v 1.1 2005/10/27 14:11:07 brun Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include "TChi2FCN.h"
#include "TChi2FitData.h"
#include "FitterUtil.h"

#include "TF1.h"
#include "TVirtualFitter.h"

#include <cassert>

//#define DEBUG 
#ifdef DEBUG
#include <iostream>
#endif

// constructor _ create FitData class

TChi2FCN::TChi2FCN( const TVirtualFitter & fitter) : 
  fUp(1), fOwner(true) 
{ 
  fFunc = dynamic_cast<TF1 *> ( fitter.GetUserFunc() );
  assert(fFunc);
  // default skip empty bins
  fData = new TChi2FitData(fitter, true); 
#ifdef DEBUG
  std::cout << "Created FitData with size = " << fData->Size() << std::endl;
#endif

  // need to set the size so ROOT can calculate ndf.
  fFunc->SetNumberFitPoints(fData->Size());
}

//  this class manages the fit data class. Delete it at the end

TChi2FCN::~TChi2FCN() {  
  if (fOwner && fData) { 
#ifdef DEBUG
    std::cout << "deleting the data - size is " << fData->Size() << std::endl; 
#endif
    delete fData; 
  }
}


  // implement chi2 function 
double TChi2FCN::operator()(const std::vector<double>& par) const {

  assert(fData); 
  assert(fFunc); 


  //  std::cout << "number of params " << par.size() << " in TF1 " << fFunc->GetNpar() << "  " << fFunc->GetNumberFreeParameters() << std::endl;
  
  unsigned int n = fData->Size();
  double chi2 = 0;
  for (unsigned int i = 0; i < n; ++ i) { 
    const std::vector<double> & x = fData->Coords(i); 
    fFunc->InitArgs( &x.front(), &par.front() ); 
    double y = fData->Value(i);
    double invError = fData->InvError(i);
    //std::cout << x[0] << "  " << y << "  " << 1./invError << " params " << par[0] << std::endl;
    double fval = 0; 
    if (fData->UseIntegral()) { 
      const std::vector<double> & x2 = fData->Coords(i+1);
      fval = FitterUtil::EvalIntegral(fFunc,x,x2,par);
    }
    else   
      fval = fFunc->EvalPar( &x.front(), &par.front() ); 

    double tmp = ( y -fval )* invError;
	  	  
    chi2 += tmp*tmp;
  }


  return chi2;
}


