/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Lightweight interface adaptor that binds a RooAbsPdf to TFOAM
// END_HTML
//


#include "RooFit.h"
#include "Riostream.h"

#include "RooTFoamBinding.h"
#include "RooRealBinding.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"

#include <assert.h>



ClassImp(RooTFoamBinding)
;


//_____________________________________________________________________________
RooTFoamBinding::RooTFoamBinding(const RooAbsReal& pdf, const RooArgSet& observables)
{
  _nset.add(observables) ;
  _binding = new RooRealBinding(pdf,observables,&_nset,kFALSE,0) ;
}


//_____________________________________________________________________________
RooTFoamBinding::~RooTFoamBinding() 
{
  // Destructor
  delete _binding ;
}



//_____________________________________________________________________________
Double_t RooTFoamBinding::Density(Int_t ndim, Double_t *xvec) 
{
  Double_t x[10] ;
  for (int i=0 ; i<ndim ; i++) {    
    x[i] = xvec[i]*(_binding->getMaxLimit(i)-_binding->getMinLimit(i)) + _binding->getMinLimit(i) ;
    //cout << "RTFB::Density xvec[" << i << "] = " << xvec[i] << " x[i] = " << x[i] << endl ;
  }
  Double_t ret = (*_binding)(x) ;  
  return ret<0?0:ret ;
}
