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
// RooMultiCategory consolidates several RooAbsCategory objects into
// a single category. The states of the multi-category consist of all the permutations
// of the input categories. 
// <p>
// RooMultiCategory state are automatically defined and updated whenever an input
// category modifies its list of states
// END_HTML
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <stdlib.h>
#include <stdio.h>
#include "TString.h"
#include "RooMultiCategory.h"
#include "RooStreamParser.h"
#include "RooArgSet.h"
#include "RooMultiCatIter.h"
#include "RooAbsCategory.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooMultiCategory)
;


//_____________________________________________________________________________
RooMultiCategory::RooMultiCategory(const char *name, const char *title, const RooArgSet& inputCatList2) :
  RooAbsCategory(name, title), _catSet("input","Input category set",this,kTRUE,kTRUE)
{  
  // Construct a product of the given set of input RooAbsCategories in 'inInputCatList'
  // The state names of this product category are {S1;S2,S3,...Sn} where Si are the state names
  // of the input categories. A RooMultiCategory is not an lvalue

  // Copy category list
  TIterator* iter = inputCatList2.createIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    if (!dynamic_cast<RooAbsCategory*>(arg)) {
      coutE(InputArguments) << "RooMultiCategory::RooMultiCategory(" << GetName() << "): input argument " << arg->GetName() 
			    << " is not a RooAbsCategory" << endl ;
    }
    _catSet.add(*arg) ;
  }
  delete iter ;
  
  updateIndexList() ;
}



//_____________________________________________________________________________
RooMultiCategory::RooMultiCategory(const RooMultiCategory& other, const char *name) :
  RooAbsCategory(other,name), _catSet("input",this,other._catSet)
{
  // Copy constructor

  updateIndexList() ;
}



//_____________________________________________________________________________
RooMultiCategory::~RooMultiCategory() 
{
  // Destructor

}



//_____________________________________________________________________________
void RooMultiCategory::updateIndexList()
{
  // Update the list of super-category states 

  // WVE broken if used with derived categories!
  clearTypes() ;

  RooMultiCatIter iter(_catSet) ;
  TObjString* obj ;
  while((obj=(TObjString*)iter.Next())) {
    // Register composite label
    defineType(obj->String()) ;
  }

  // Renumbering will invalidate cache
  setValueDirty() ;
}



//_____________________________________________________________________________
TString RooMultiCategory::currentLabel() const
{
  // Return the name of the current state, 
  // constructed from the state names of the input categories

  TIterator* lIter = _catSet.createIterator() ;

  // Construct composite label name
  TString label ;
  RooAbsCategory* cat ;
  Bool_t first(kTRUE) ;
  while((cat=(RooAbsCategory*) lIter->Next())) {
    label.Append(first?"{":";") ;
    label.Append(cat->getLabel()) ;      
    first=kFALSE ;
  }
  label.Append("}") ;  
  delete lIter ;

  return label ;
}



//_____________________________________________________________________________
RooCatType RooMultiCategory::evaluate() const
{
  // Calculate the current value 

  if (isShapeDirty()) const_cast<RooMultiCategory*>(this)->updateIndexList() ;

  // current label is can be looked up by definition 
  // coverity[NULL_RETURNS] 
  return *lookupType(currentLabel()) ;
}



//_____________________________________________________________________________
void RooMultiCategory::printMultiline(ostream& os, Int_t content, Bool_t verbose, TString indent) const
{
  // Print the state of this object to the specified output stream.

  RooAbsCategory::printMultiline(os,content,verbose,indent) ;
  
  if (verbose) {     
    os << indent << "--- RooMultiCategory ---" << endl;
    os << indent << "  Input category list:" << endl ;
    TString moreIndent(indent) ;
    moreIndent.Append("   ") ;
    _catSet.printStream(os,kName|kValue,kStandard,moreIndent.Data()) ;
  }
}



//_____________________________________________________________________________
Bool_t RooMultiCategory::readFromStream(istream& /*is*/, Bool_t /*compact*/, Bool_t /*verbose*/) 
{
  // Read object contents from given stream
  return kTRUE ;
}



//_____________________________________________________________________________
void RooMultiCategory::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream
  RooAbsCategory::writeToStream(os,compact) ;
}
