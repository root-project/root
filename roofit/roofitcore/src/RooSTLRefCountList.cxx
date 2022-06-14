// Author: Stephan Hageboeck, CERN, 12/2018
/*****************************************************************************
 * Project: RooFit                                                           *
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

#include "RooSTLRefCountList.h"

#include "RooRefCountList.h"
#include "RooLinkedListIter.h"
#include "RooAbsArg.h"
#include <string>

// Template specialisation used in RooAbsArg:
ClassImp(RooSTLRefCountList<RooAbsArg>);

namespace RooFit {
namespace STLRefCountListHelpers {
/// Converts RooRefCountList to RooSTLRefCountList<RooAbsArg>.
/// This converter only yields lists with T=RooAbsArg. This is ok because this
/// the old RefCountList was only holding these.
RooSTLRefCountList<RooAbsArg> convert(const RooRefCountList& old) {
  RooSTLRefCountList<RooAbsArg> newList;
  newList.reserve(old.GetSize());

  auto it = old.fwdIterator();
  for (RooAbsArg * elm = it.next(); elm != nullptr; elm = it.next()) {
    newList.Add(elm, old.refCount(elm));
  }

  return newList;
}
}
}

