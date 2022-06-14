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

/**
\file RooTFoamBinding.cxx
\class RooTFoamBinding
\ingroup Roofitcore

Lightweight interface adaptor that binds a RooAbsPdf to TFOAM
**/


#include "Riostream.h"

#include "RooTFoamBinding.h"
#include "RooRealBinding.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"

#include <assert.h>



using namespace std;

ClassImp(RooTFoamBinding);
;


////////////////////////////////////////////////////////////////////////////////

RooTFoamBinding::RooTFoamBinding(const RooAbsReal& pdf, const RooArgSet& observables)
{
  _nset.add(observables) ;
  _binding = new RooRealBinding(pdf,observables,&_nset,false,0) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooTFoamBinding::~RooTFoamBinding()
{
  delete _binding ;
}



////////////////////////////////////////////////////////////////////////////////

double RooTFoamBinding::Density(Int_t ndim, double *xvec)
{
  double x[10] ;
  for (int i=0 ; i<ndim ; i++) {
    x[i] = xvec[i]*(_binding->getMaxLimit(i)-_binding->getMinLimit(i)) + _binding->getMinLimit(i) ;
    //cout << "RTFB::Density xvec[" << i << "] = " << xvec[i] << " x[i] = " << x[i] << endl ;
  }
  double ret = (*_binding)(x) ;
  return ret<0?0:ret ;
}
