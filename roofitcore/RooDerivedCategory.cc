/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include <stdlib.h>
#include "TString.h"
#include "TH1.h"
#include "RooFitCore/RooDerivedCategory.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/Roo1DTable.hh"

ClassImp(RooDerivedCategory) 
;


RooDerivedCategory::RooDerivedCategory(const char *name, const char *title) : 
  RooAbsCategory(name,title)
{
}



RooDerivedCategory::RooDerivedCategory(const char* name, const RooDerivedCategory& other) :
  RooAbsCategory(name, other)
{
}



RooDerivedCategory::RooDerivedCategory(const RooDerivedCategory& other) : 
  RooAbsCategory(other)
{
}


RooDerivedCategory::~RooDerivedCategory()
{
}


RooDerivedCategory& RooDerivedCategory::operator=(const RooDerivedCategory& other)
{
  RooAbsCategory::operator=(other) ;
  return *this ;
}


RooAbsArg& RooDerivedCategory::operator=(const RooAbsArg& aother)
{
  return operator=((const RooDerivedCategory&)aother) ;
}



Int_t RooDerivedCategory::getIndex() const
{
  if (isValueDirty() || isShapeDirty()) {
    _value = traceEval() ;

    setValueDirty(false) ;
    setShapeDirty(false) ;
  } 

  return _value.getVal() ;
}


const char* RooDerivedCategory::getLabel() const
{
  if (isValueDirty() || isShapeDirty()) {
    _value = traceEval() ;

    setValueDirty(false) ;
    setShapeDirty(false) ;
  } 

  return _value.GetName() ;
}




RooCatType RooDerivedCategory::traceEval() const
{
  RooCatType value = evaluate() ;
  
  //Standard tracing code goes here
  if (!isValid(value)) {
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}

