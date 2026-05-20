/// \file ROOT/RNTupleClassicBrowse.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2025-07-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleBrowseUtils
#define ROOT_RNTupleBrowseUtils

#include <ROOT/RNTupleTypes.hxx>

namespace ROOT {

class RNTupleDescriptor;

namespace Internal {

// Skips "internal" sub fields that should not appear in the field tree in the browser because
// they clutter the view, e.g. the _0 subfields of vectors. The return value is either fieldId if
// there is nothing to skip or it is a subfield of fieldId. Skipping of fields is applied recursively,
// e.g. for fieldId representing a vector<vector<float>>, two levels are skipped in the field hierarchy.
DescriptorId_t GetNextBrowsableField(DescriptorId_t fieldId, const RNTupleDescriptor &desc);

} // namespace Internal
} // namespace ROOT

#endif
