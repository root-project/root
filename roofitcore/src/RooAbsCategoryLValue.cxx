/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsCategoryLValue.cc,v 1.1 2001/05/10 00:16:06 verkerke Exp $
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
#include <string.h>
#include "TTree.h"
#include "TString.h"
#include "TH1.h"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooAbsCategoryLValue) 
;


RooAbsCategoryLValue::RooAbsCategoryLValue(const char *name, const char *title) : 
  RooAbsCategory(name,title)
{
  setValueDirty(kTRUE) ;  
  setShapeDirty(kTRUE) ;  
}


RooAbsCategoryLValue::RooAbsCategoryLValue(const RooAbsCategoryLValue& other, const char* name) :
  RooAbsCategory(other, name)
{
}


RooAbsCategoryLValue::~RooAbsCategoryLValue()
{
}


RooAbsCategoryLValue& RooAbsCategoryLValue::operator=(Int_t index) {
  setIndex(index,kTRUE) ;
  return *this ;
}


RooAbsCategoryLValue& RooAbsCategoryLValue::operator=(const char*label) {
  setLabel(label) ;
  return *this ;
}



Bool_t RooAbsCategoryLValue::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream
}



void RooAbsCategoryLValue::writeToStream(ostream& os, Bool_t compact) const
{
}






