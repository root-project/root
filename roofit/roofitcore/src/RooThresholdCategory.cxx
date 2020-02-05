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
\file RooThresholdCategory.cxx
\class RooThresholdCategory
\ingroup Roofitcore

Class RooThresholdCategory provides a real-to-category mapping defined
by a series of thresholds.
**/


#include "RooThresholdCategory.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooThresholdCategory);

namespace {
bool threshListSorter(const std::pair<double,RooCatType>& lhs, const std::pair<double,RooCatType>& rhs) {
  return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.second.getVal() < rhs.second.getVal());
}
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with input function to be mapped and name and index of default
/// output state of unmapped values

RooThresholdCategory::RooThresholdCategory(const char *name, const char *title, RooAbsReal& inputVar, 
					   const char* defOut, Int_t defIdx) :
  RooAbsCategory(name, title), _inputVar("inputVar","Input category",this,inputVar)
{
  _defCat = defineType(defOut,defIdx);
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooThresholdCategory::RooThresholdCategory(const RooThresholdCategory& other, const char *name) :
  RooAbsCategory(other,name), _inputVar("inputVar",this,other._inputVar)
{
  _defCat = lookupType(other._defCat->GetName());

  for (const auto& cat : other._threshList){
    _threshList.push_back(cat);
  }
  std::sort(_threshList.begin(), _threshList.end(), threshListSorter);
}


////////////////////////////////////////////////////////////////////////////////
/// Insert threshold at value upperLimit. All values below upper limit (and above any lower
/// thresholds, if any) will be mapped to a state name 'catName' with index 'catIdx'

Bool_t RooThresholdCategory::addThreshold(Double_t upperLimit, const char* catName, Int_t catIdx) 
{  
  // Check if identical threshold values is not defined yet
  for (const auto& thresh : _threshList) {
    if (thresh.first == upperLimit) {
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

  _threshList.emplace_back(upperLimit, *type);
  std::sort(_threshList.begin(), _threshList.end(), threshListSorter);
     
  return kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate and return the value of the mapping function

RooCatType RooThresholdCategory::evaluate() const
{
  // Scan the threshold list
  for (const auto& thresh : _threshList) {
    if (_inputVar<thresh.first)
      return thresh.second;
  }

  // Return default if nothing found
  return *_defCat;
}



////////////////////////////////////////////////////////////////////////////////
/// Write object contents to given stream

void RooThresholdCategory::writeToStream(ostream& os, Bool_t compact) const
{
  if (compact) {
    // Write value only
    os << getLabel() ;
  } else {
    // Write mapping expression

    // Scan list of threshold
    for (const auto& thresh : _threshList) {
      os << thresh.second.GetName() << ":<" << thresh.first << " " ;
    }
    os << _defCat->GetName() << ":*" ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Print info about this threshold category to the specified stream. In addition to the info
/// from RooAbsCategory::printStream() we add:
///
///  Standard : input category
///     Shape : default value
///   Verbose : list of thresholds

void RooThresholdCategory::printMultiline(ostream& os, Int_t content, Bool_t verbose, TString indent) const
{
   RooAbsCategory::printMultiline(os,content,verbose,indent);

   if (verbose) {
     os << indent << "--- RooThresholdCategory ---" << endl
	<< indent << "  Maps from " ;
     _inputVar.arg().printStream(os,0,kStandard);
     
     os << indent << "  Threshold list" << endl ;
     for (const auto& thresh : _threshList) {
       os << indent << "    input < " << thresh.first << " --> " ;
       thresh.second.printStream(os,kName|kValue,kSingleLine) ;
     }
     os << indent << "  Default value is " ;
     _defCat->printStream(os,kValue,kSingleLine);


   }
}


