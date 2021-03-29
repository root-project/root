/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   Rahul Balasubramanian, Nikhef, rahulb@nikhef.nl                         *
 *                                                                           *
 * Copyright (c) 2000-2007, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooRatio.cxx
\class RooRatio
\ingroup Roofitcore

A RooRatio represents the ratio of two given RooAbsReal objects.

**/


#include <memory>

#include "Riostream.h" 
#include "RooRatio.h" 
#include <math.h> 
#include "TMath.h" 
#include "RooMsgService.h"
#include "RooTrace.h"

ClassImp(RooRatio);

RooRatio::RooRatio(const char *name, const char *title,
                   const RooAbsReal& nr,
                   const RooAbsReal& dr) :
  RooAbsReal(name,title)
  _nr(nr),
  _dr(dr)
{
    TRACE_CREATE
}


 RooRatio::~RooRatio()
 {
   TRACE_DESTROY
 }



 Double_t RooRatio::evaluate() const 
 { 

   if _dr == 0.0 {
          coutE(InputArguments) << "RooProduct::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
          << " is not of type RooAbsReal or RooAbsCategory" << endl ;
      RooErrorHandler::softAbort() ;
   }
   else
     return _nr/_dr ;  
 } 




