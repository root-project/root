/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UCSB, verkerke@slac.stanford.edu
 * History:
 *   01-Mar-2001 WV Create initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include "TString.h"
#include "RooFitCore/RooSuperCategory.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooMultiCatIter.hh"

ClassImp(RooSuperCategory)


RooSuperCategory::RooSuperCategory(const char *name, const char *title, RooArgSet& inputCatList) :
  RooAbsCategoryLValue(name, title) 
{  
  // Copy category list
  TIterator* iter = inputCatList.MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    _catList.add(*arg) ;
    addServer(*arg,kTRUE,kTRUE) ;
  }
  delete iter ;

  updateIndexList() ;
}


RooSuperCategory::RooSuperCategory(const RooSuperCategory& other, const char *name) :
  RooAbsCategoryLValue(other,name)
{
  TIterator* iter = other._catList.MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    _catList.add(*arg) ;
  }
  delete iter ;

  updateIndexList() ;
}



RooSuperCategory::~RooSuperCategory() 
{
}


void RooSuperCategory::updateIndexList()
{
  clearTypes() ;
  RooArgSet catListClone("catListClone",_catList) ;
  RooMultiCatIter mcIter(_catList) ;

  while(mcIter.Next()) {
    // Register composite label
    defineType(currentLabel()) ;
  }

  // Restore original input state
  _catList = catListClone ;

  // Renumbering will invalidate cache
  setValueDirty(kTRUE) ;
}


TString RooSuperCategory::currentLabel() const
{
  TIterator* lIter = _catList.MakeIterator() ;

  // Construct composite label name
  TString label ;
  RooAbsCategory* cat ;
  Bool_t first(kTRUE) ;
  while(cat=(RooAbsCategory*) lIter->Next()) {
    label.Append(first?"{":";") ;
    label.Append(cat->getLabel()) ;      
    first=kFALSE ;
  }
  label.Append("}") ;  
  delete lIter ;

  return label ;
}


RooCatType
RooSuperCategory::evaluate() const
{
  if (isShapeDirty()) updateIndexList() ;
  return *lookupType(currentLabel()) ;
}


Bool_t RooSuperCategory::setIndex(Int_t index, Bool_t printError) 
{
}


Bool_t RooSuperCategory::setLabel(const char* label, Bool_t printError) 
{
}


void RooSuperCategory::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  RooAbsCategory::printToStream(os,opt,indent) ;
}


Bool_t RooSuperCategory::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  return kTRUE ;
}



void RooSuperCategory::writeToStream(ostream& os, Bool_t compact) const
{
  RooAbsCategory::writeToStream(os,compact) ;
}
