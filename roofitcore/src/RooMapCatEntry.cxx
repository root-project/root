/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooMapCatEntry.cc,v 1.1 2001/05/10 00:16:07 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooMapCatEntry is an auxilary class for RooMappedCategory and defines a 
// a mapping. A mapping consists of a wildcard regular expression, that
// can be matched against the input category state label and an output
// category state (RooCatType), which should be assign if the match is successfull.

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
