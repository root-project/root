/// \file ROOT/RNTupleClassicBrowse.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2025-06-60

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleClassicBrowse
#define ROOT_RNTupleClassicBrowse

class TBrowser;

namespace ROOT {
namespace Internal {

void BrowseRNTuple(const void *ntuple, TBrowser *b);

} // namespace Internal
} // namespace ROOT

#endif
