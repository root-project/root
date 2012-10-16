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
// Class RooThresholdCategory provides a real-to-category mapping defined
// by a series of thresholds.
// END_HTML
//


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <stdlib.h>
#include <stdio.h>
#include "TString.h"
#include "RooThresholdCategory.h"
#include "RooStreamParser.h"
#include "RooThreshEntry.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooThresholdCategory)



//_____________________________________________________________________________
RooThresholdCategory::RooThresholdCategory(const char *name, const char *title, RooAbsReal& inputVar, 
					   const char* defOut, Int_t defIdx) :
  RooAbsCategory(name, title), _inputVar("inputVar","Input category",this,inputVar)
{
  // Constructor with input function to be mapped and name and index of default
  // output state of unmapped values

  _defCat = (RooCatType*) defineType(defOut,defIdx) ;
  _threshIter = _threshList.MakeIterator() ;
}



//_____________________________________________________________________________
RooThresholdCategory::RooThresholdCategory(const RooThresholdCategory& other, const char *name) :
  RooAbsCategory(other,name), _inputVar("inputVar",this,other._inputVar)
{
  // Copy constructor

  _defCat = (RooCatType*) lookupType(other._defCat->GetName()) ;

  other._threshIter->Reset() ;
  RooThreshEntry* te ;
  while((te=(RooThreshEntry*)other._threshIter->Next())) {
    _threshList.Add(new RooThreshEntry(*te)) ;
  }

  _threshIter = _threshList.MakeIterator() ;
}



//_____________________________________________________________________________
RooThresholdCategory::~RooThresholdCategory() 
{
  // Destructor

  _threshList.Delete() ;
  delete _threshIter ;
}



//_____________________________________________________________________________
Bool_t RooThresholdCategory::addThreshold(Double_t upperLimit, const char* catName, Int_t catIdx) 
{  
  // Insert threshold at value upperLimit. All values below upper limit (and above any lower
  // thresholds, if any) will be mapped to a state name 'catName' with index 'catIdx'

  // Check if identical threshold values is not defined yet
  _threshIter->Reset() ;
  RooThreshEntry* te ;
  while ((te=(RooThreshEntry*)_threshIter->Next())) {
    if (te->thresh() == upperLimit) {
      coutW(InputArguments) << "RooThresholdCategory::addThreshold(" << GetName() 
			    << ") threshold at " << upperLimit << " already defined" << endl ;
      return kTRUE ;
    }    
  }


  // Add a threshold entry
  const RooCatType* type = lookupType(catName,kFALSE) ;
  if (!type) {
    if (catIdx==-99999) {
      type=defineType(catName) ;
    } else {
      type=defineType(catName,catIdx) ;      
    }
  }
  te = new RooThreshEntry(upperLimit,*type) ;
  _threshList.Add(te) ;
     
  return kFALSE ;
}



//_____________________________________________________________________________
RooCatType RooThresholdCategory::evaluate() const
{
  // Calculate and return the value of the mapping function

  // Scan the threshold list
  _threshIter->Reset() ;
  RooThreshEntry* te ;
  while((te=(RooThreshEntry*)_threshIter->Next())) {
    if (_inputVar<te->thresh()) return te->cat() ;
  }

  // Return default if nothing found
  return *_defCat ;
}



//_____________________________________________________________________________
void RooThresholdCategory::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream

  if (compact) {
    // Write value only
    os << getLabel() ;
  } else {
    // Write mapping expression

    // Scan list of threshold
    _threshIter->Reset() ;
    RooThreshEntry* te ;
    while((te=(RooThreshEntry*)_threshIter->Next())) {
      os << te->cat().GetName() << ":<" << te->thresh() << " " ;
    }
    os << _defCat->GetName() << ":*" ;
  }
}



//_____________________________________________________________________________
void RooThresholdCategory::printMultiline(ostream& os, Int_t content, Bool_t verbose, TString indent) const
{
  // Print info about this threshold category to the specified stream. In addition to the info
  // from RooAbsCategory::printStream() we add:
  //
  //  Standard : input category
  //     Shape : default value
  //   Verbose : list of thresholds

   RooAbsCategory::printMultiline(os,content,verbose,indent);

   if (verbose) {
     os << indent << "--- RooThresholdCategory ---" << endl
	<< indent << "  Maps from " ;
     _inputVar.arg().printStream(os,0,kStandard);
     
     os << indent << "  Threshold list" << endl ;
     _threshIter->Reset() ;
     RooThreshEntry* te ;
     while((te=(RooThreshEntry*)_threshIter->Next())) {
       os << indent << "    input < " << te->thresh() << " --> " ; 
       te->cat().printStream(os,kName|kValue,kSingleLine) ;
     }
     os << indent << "  Default value is " ;
     _defCat->printStream(os,kValue,kSingleLine);


   }
}


