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
// RooNameSet is a utility class that stores the names the objects
// in a RooArget. This allows to preserve the contents of a RooArgSet
// in a specific use contents beyond the lifespan of the object in
// the RooArgSet. A new RooArgSet can be created from a RooNameSet
// by offering it a list of new RooAbsArg objects. 
// END_HTML
//

#include <cstring>
#include <algorithm>
#include <cassert>

#include "RooFit.h"
#include "Riostream.h"

#include "TObjString.h"
#include "TClass.h"
#include "RooNameSet.h"
#include "RooArgSet.h"
#include "RooArgList.h"

ClassImp(RooNameSet)
;

//_____________________________________________________________________________
void RooNameSet::strdup(Int_t& dstlen, char* &dstbuf, const char* src)
{
  // copy src to dst, keep dstlen up to date, make sure zero length strings
  // do not take memory
  dstlen = src ? std::strlen(src) : 0;
  if (dstlen) ++dstlen;
  char *buf = dstlen ? new char[dstlen] : 0;
  if (buf) std::strcpy(buf, src);
  delete[] dstbuf;
  dstbuf = buf;
}

//_____________________________________________________________________________
RooNameSet::RooNameSet() : _len(0), _nameList(0)
{
  // Default constructor
}

//_____________________________________________________________________________
RooNameSet::RooNameSet(const RooArgSet& argSet) : _len(0), _nameList(0)
{
  // Construct from RooArgSet
  refill(argSet);
}

//_____________________________________________________________________________
RooNameSet::RooNameSet(const RooNameSet& other) :
  TObject(other), RooPrintable(other), _len(0), _nameList(0)
{
  // Copy constructor
  strdup(_len, _nameList, other._nameList);
}

//_____________________________________________________________________________
RooNameSet::~RooNameSet() 
{
  // Destructor
  delete[] _nameList;
}

//_____________________________________________________________________________
RooNameSet& RooNameSet::operator=(const RooNameSet& other) 
{
  // Assignment operator

  // Check comparison against self
  if (&other == this || _nameList == other._nameList) return *this;

  strdup(_len, _nameList, other._nameList);

  return *this;
}

//_____________________________________________________________________________
Bool_t RooNameSet::operator==(const RooNameSet& other) const
{
  // Comparison operator

  // Check comparison against self
  if (&other == this || _nameList == other._nameList) return kTRUE;

  return _nameList && other._nameList &&
    0 == std::strcmp(_nameList, other._nameList);
}

//_____________________________________________________________________________
Bool_t RooNameSet::operator<(const RooNameSet& other) const 
{
  if (&other == this) return kFALSE;
  if (!_nameList) return other._nameList;
  if (!other._nameList) return kFALSE;
  return std::strcmp(_nameList, other._nameList) < 0;
}

//_____________________________________________________________________________
void RooNameSet::extendBuffer(Int_t inc)
{
  if (!inc) return;
  assert(inc > 0 || _len >= -inc);
  int newsz = _len + inc;
  if (newsz <= 1 || !_len) newsz = 0;
  char* newbuf = newsz ? new char[newsz] : 0;
  if (newbuf && _nameList) {
    std::strncpy(newbuf, _nameList, std::min(_len, newsz));
    newbuf[newsz - 1] = 0;
  }
  delete[] _nameList;
  _nameList = newbuf;
  _len = newsz;
}

//_____________________________________________________________________________
void RooNameSet::setNameList(const char* givenList) 
{
  strdup(_len, _nameList, givenList);
}

//_____________________________________________________________________________
void RooNameSet::refill(const RooArgSet& argSet) 
{
  // Refill internal contents from names in given argSet
  delete[] _nameList;
  _nameList = 0;
  _len = 0;
  if (0 == argSet.getSize()) return;

  RooArgList tmp(argSet);
  tmp.sort();
  // figure out the length of the array we need
  RooAbsArg* arg = 0;
  for (RooFIter it = tmp.fwdIterator(); 0 != (arg = it.next());
      _len += 1 + std::strlen(arg->GetName())) { }
  if (_len <= 1) _len = 0;
  // allocate it
  _nameList = _len ? new char[_len] : 0;
  if (_nameList) {
    // copy in the names of the objects
    char *p = _nameList;
    for (RooFIter it = tmp.fwdIterator(); 0 != (arg = it.next()); ) {
      const char *name = arg->GetName();
      std::strcpy(p, name);
      while (*p) ++p;
      *p++ = ':';
    }
    // zero-terminate properly
    *--p = 0;
  }
}

//_____________________________________________________________________________
RooArgSet* RooNameSet::select(const RooArgSet& list) const 
{
  // Construct a RooArgSet of objects in input 'list'
  // whose names match to those in the internal name
  // list of RooNameSet

  RooArgSet* output = new RooArgSet;
  if (!_nameList || !std::strlen(_nameList)) return output;

  // need to copy _nameList because std::strtok modifies the string
  char* tmp = 0;
  int dummy = 0;
  strdup(dummy, tmp, _nameList);

  char* token = std::strtok(tmp, ":"); 
  while (token) {
    RooAbsArg* arg = list.find(token);
    if (arg) output->add(*arg);
    token = std::strtok(0, ":");
  }
  delete[] tmp;

  return output;
}

//_____________________________________________________________________________
void RooNameSet::printName(std::ostream& os) const 
{
  // Print name of nameset
  os << GetName();
}

//_____________________________________________________________________________
void RooNameSet::printTitle(std::ostream& os) const 
{
  // Print title of nameset
  os << GetTitle();
}

//_____________________________________________________________________________
void RooNameSet::printClassName(std::ostream& os) const 
{
  // Print class name of nameset
  os << IsA()->GetName();
}

//_____________________________________________________________________________
void RooNameSet::printValue(std::ostream& os) const 
{
  // Print value of nameset, i.e the list of names
  os << content();
}
