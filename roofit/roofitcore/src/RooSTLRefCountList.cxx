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

/// \cond ROOFIT_INTERNAL

#include "RooSTLRefCountList.h"

#include "RooLinkedList.h"
#include "RooLinkedListIter.h"
#include "RooAbsArg.h"

// Template specialisation used in RooAbsArg:

/// Converts RooLinkedList to RooSTLRefCountList<RooAbsArg>.
/// This converter only yields lists with T=RooAbsArg. This is ok because this
/// the old RefCountList was only holding these.
template <>
RooSTLRefCountList<RooAbsArg> RooSTLRefCountList<RooAbsArg>::convert(const RooLinkedList& old) {
  RooSTLRefCountList<RooAbsArg> newList;
  newList.reserve(old.GetSize());

  for(TObject * elm : old) {
    newList.Add(static_cast<RooAbsArg*>(elm), old.findLink(elm)->refCount());
  }

  return newList;
}

/// \endcond
