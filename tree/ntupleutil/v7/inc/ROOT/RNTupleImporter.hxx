/// \file ROOT/RNTuplerImporter.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2022-11-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTuplerImporter
#define ROOT7_RNTuplerImporter

#include <ROOT/RStringView.hxx>

#include <memory>

class TTree;

namespace ROOT {
namespace Experimental {

namespace Detail {
class RPageSink;
}

// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleImporter
\ingroup NTuple
\brief Converts a TTree into an RNTuple

The class steers the conversion of a TTree into an RNTuple.
*/
// clang-format on
class RNTupleImporter {
private:
   RNTupleImporter() = default;

public:
   RNTupleImporter(const RNTupleImporter &other) = delete;
   RNTupleImporter &operator=(const RNTupleImporter &other) = delete;
   RNTupleImporter(RNTupleImporter &&other) = delete;
   RNTupleImporter &operator=(RNTupleImporter &&other) = delete;
   ~RNTupleImporter() = default;

   static std::unique_ptr<RNTupleImporter>
   Create(std::string_view sourceFile, std::string_view treeName, std::string_view destFile);
}; // class RNTupleImporter

} // namespace Experimental
} // namespace ROOT

#endif
