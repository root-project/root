/// \file ROOT/RNTupleTreeMap.hxx
/// \ingroup NTuple
/// \author Patryk Tymoteusz Pilichowski <patryk.tymoteusz.pilichowski@cern.ch>
/// \date 2025-09-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleTreeMap
#define ROOT_RNTupleTreeMap

#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RTreeMapPainter.hxx>

#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {
/////////////////////////////////////////////////////////////////////////////
/// \brief Logic for converting an RNTuple to RTreeMapPainter given RNTupleInspector
std::unique_ptr<RTreeMapPainter> CreateTreeMapFromRNTuple(const RNTupleInspector &insp);

/////////////////////////////////////////////////////////////////////////////
/// \brief Logic for converting an RNTuple to RTreeMapPainter given file and tuple names
std::unique_ptr<RTreeMapPainter> CreateTreeMapFromRNTuple(std::string_view sourceFileName, std::string_view tupleName);
} // namespace Experimental

} // namespace ROOT

#endif
