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
// RooAbsMCStudyModule is a base class for add-on modules to RooMCStudy that
// can perform additional calculations on each generate+fit cycle managed
// by RooMCStudy
//
// This class can insert code to be executed before each generation step,
// between the generation and fitting step and after the fitting step.
// Any summary output variables declared in the RooDataSet exported through
// summaryData() is merged with the 'master' summary dataset in RooMCStudy
//
// Look at RooDLLSignificanceMCStudyModule for an example of an implementation
// END_HTML
//
//

#include "RooFit.h"
#include "RooAbsMCStudyModule.h"

using namespace std;

ClassImp(RooAbsMCStudyModule)
  ;


//_____________________________________________________________________________
RooAbsMCStudyModule::RooAbsMCStudyModule(const char* name, const char* title) : TNamed(name,title), _mcs(0) 
{
  // Constructor
} 


//_____________________________________________________________________________
RooAbsMCStudyModule::RooAbsMCStudyModule(const RooAbsMCStudyModule& other) : TNamed(other), _mcs(other._mcs)
{
  // Copy constructor
} 


//_____________________________________________________________________________
Bool_t RooAbsMCStudyModule::doInitializeInstance(RooMCStudy& study) 
{ 
  // Store reference to RooMCStudy object that this module relates to and call internal module
  // initialization function
  _mcs = &study ; 
  return initializeInstance() ; 
}  

