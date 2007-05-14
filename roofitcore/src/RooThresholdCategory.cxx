/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooThresholdCategory.cxx,v 1.14 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [CAT] --
// RooThresholdCategory provides a real-to-category mapping defined
// by a series of thresholds.


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <stdlib.h>
#include <stdio.h>
#include "TString.h"
#include "RooThresholdCategory.h"
#include "RooStreamParser.h"
#include "RooThreshEntry.h"

ClassImp(RooThresholdCategory)


RooThresholdCategory::RooThresholdCategory(const char *name, const char *title, RooAbsReal& inputVar, 
					   const char* defOut, Int_t defIdx) :
  RooAbsCategory(name, title), _inputVar("inputVar","Input category",this,inputVar)
{
  // Constructor with input category and name of default (unmapped) output state
  _defCat = (RooCatType*) defineType(defOut,defIdx) ;
  _threshIter = _threshList.MakeIterator() ;
}


RooThresholdCategory::RooThresholdCategory(const RooThresholdCategory& other, const char *name) :
  RooAbsCategory(other,name), _inputVar("inputVar",this,other._inputVar)
{
  _defCat = (RooCatType*) lookupType(other._defCat->GetName()) ;

  // Copy constructor
  other._threshIter->Reset() ;
  RooThreshEntry* te ;
  while((te=(RooThreshEntry*)other._threshIter->Next())) {
    _threshList.Add(new RooThreshEntry(*te)) ;
  }

  _threshIter = _threshList.MakeIterator() ;
}



RooThresholdCategory::~RooThresholdCategory() 
{
  // Destructor
  _threshList.Delete() ;
  delete _threshIter ;
}



Bool_t RooThresholdCategory::addThreshold(Double_t upperLimit, const char* catName, Int_t catIdx) 
{  
  // Check if identical threshold values is not defined yet
  _threshIter->Reset() ;
  RooThreshEntry* te ;
  while ((te=(RooThreshEntry*)_threshIter->Next())) {
    if (te->thresh() == upperLimit) {
      cout << "RooThresholdCategory::addThreshold(" << GetName() 
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



RooCatType
RooThresholdCategory::evaluate() const
{
  // Scan the threshold list
  _threshIter->Reset() ;
  RooThreshEntry* te ;
  while((te=(RooThreshEntry*)_threshIter->Next())) {
    if (_inputVar<te->thresh()) return te->cat() ;
  }

  // Return default if nothing found
  return *_defCat ;
}



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


void RooThresholdCategory::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this threshold category to the specified stream. In addition to the info
  // from RooAbsCategory::printToStream() we add:
  //
  //  Standard : input category
  //     Shape : default value
  //   Verbose : list of thresholds

   RooAbsCategory::printToStream(os,opt,indent);

   if (opt > Standard) {
     os << indent << "--- RooThresholdCategory ---" << endl
	<< indent << "  Maps from " ;
     _inputVar.arg().printToStream(os,Standard);
     
     os << indent << "  Threshold list" << endl ;
     _threshIter->Reset() ;
     RooThreshEntry* te ;
     while((te=(RooThreshEntry*)_threshIter->Next())) {
       os << indent << "    input < " << te->thresh() << " --> " ; 
       te->cat().printToStream(os,OneLine) ;
     }
     os << indent << "  Default value is ";
     _defCat->printToStream(os,OneLine);


   }
}


