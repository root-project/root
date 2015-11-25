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
\file RooThreshEntry.cxx
\class RooThreshEntry
\ingroup Roofitcore

Class RooThreshEntry is a utility class for RooThresholdCategory
**/

#include "RooFit.h"

#include "TClass.h"
#include "TClass.h"
#include "RooThreshEntry.h"

using namespace std;

ClassImp(RooThreshEntry)
;


////////////////////////////////////////////////////////////////////////////////
/// Constructor with threshold value and associated category value

RooThreshEntry::RooThreshEntry(Double_t inThresh, const RooCatType& inCat) : 
  _thresh(inThresh), _cat(inCat) 
{
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooThreshEntry::RooThreshEntry(const RooThreshEntry& other) : 
  TObject(other), _thresh(other._thresh), _cat(other._cat) 
{
}



////////////////////////////////////////////////////////////////////////////////
/// Implement TObject Compare() operator to facilitate sorting of 
/// RooThreshEntry object in ROOT collection. Only comparison
/// to other RooThreshEntry objects is supported

Int_t RooThreshEntry::Compare(const TObject* other) const 
{
  // Can only compare objects of same type
  if (!other->IsA()->InheritsFrom(RooThreshEntry::Class())) return 0 ;

  RooThreshEntry* otherTE = (RooThreshEntry*) other ;
  return (_thresh < otherTE->_thresh) ? -1 : 1 ;
}


