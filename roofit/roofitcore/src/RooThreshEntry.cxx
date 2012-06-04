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
// Class RooThreshEntry is a utility class for RooThresholdCategory
// END_HTML
//

#include "RooFit.h"

#include "TClass.h"
#include "TClass.h"
#include "RooThreshEntry.h"

using namespace std;

ClassImp(RooThreshEntry)
;


//_____________________________________________________________________________
RooThreshEntry::RooThreshEntry(Double_t inThresh, const RooCatType& inCat) : 
  _thresh(inThresh), _cat(inCat) 
{
  // Constructor with threshold value and associated category value
}



//_____________________________________________________________________________
RooThreshEntry::RooThreshEntry(const RooThreshEntry& other) : 
  TObject(other), _thresh(other._thresh), _cat(other._cat) 
{
  // Copy constructor
}



//_____________________________________________________________________________
Int_t RooThreshEntry::Compare(const TObject* other) const 
{
  // Implement TObject Compare() operator to facilitate sorting of 
  // RooThreshEntry object in ROOT collection. Only comparison
  // to other RooThreshEntry objects is supported

  // Can only compare objects of same type
  if (!other->IsA()->InheritsFrom(RooThreshEntry::Class())) return 0 ;

  RooThreshEntry* otherTE = (RooThreshEntry*) other ;
  return (_thresh < otherTE->_thresh) ? -1 : 1 ;
}


