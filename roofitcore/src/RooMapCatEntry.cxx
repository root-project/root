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

#include "RooFitCore/RooMapCatEntry.hh"

ClassImp(RooMapCatEntry)
;

RooMapCatEntry::RooMapCatEntry(const char* exp, const RooCatType* cat) : 
  TNamed(exp,exp), _regexp(exp,kTRUE), _cat((RooCatType*)cat) 
{
}


RooMapCatEntry::RooMapCatEntry(const RooMapCatEntry& other) : 
  TNamed(other), _regexp(other.GetName(),kTRUE), _cat(other._cat) 
{
}


Bool_t RooMapCatEntry::match(const char* testPattern) const 
{
  return (TString(testPattern).Index(_regexp)>=0) ;
}
