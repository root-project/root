/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooThreshEntry.cc,v 1.1 2001/07/31 05:54:22 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --

#include "TClass.h"
#include "RooFitCore/RooThreshEntry.hh"

ClassImp(RooThreshEntry)
;

RooThreshEntry::RooThreshEntry(Double_t thresh, const RooCatType& cat) : 
  _thresh(thresh), _cat(cat) 
{
}


RooThreshEntry::RooThreshEntry(const RooThreshEntry& other) : 
  TObject(other), _thresh(other._thresh), _cat(other._cat) 
{
}


Int_t RooThreshEntry::Compare(const TObject* other) const 
{
  // Can only compare objects of same type
  if (!other->IsA()->InheritsFrom(RooThreshEntry::Class())) return 0 ;

  RooThreshEntry* otherTE = (RooThreshEntry*) other ;
  return (_thresh < otherTE->_thresh) ? -1 : 1 ;
}


