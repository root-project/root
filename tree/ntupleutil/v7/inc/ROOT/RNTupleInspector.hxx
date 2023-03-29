/// \file ROOT/RNTuplerInspector.hxx
/// \ingroup NTuple ROOT7
/// \author Florine de Geus <florine.willemijn.de.geus@cern.ch>
/// \date 2023-01-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleInspector
#define ROOT7_RNTupleInspector

#include <ROOT/RError.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>

#include <cstdlib>
#include <memory>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleInspector
\ingroup NTuple
\brief Inspect a given RNTuple

Example usage:

~~~ {.cpp}
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleInspector.hxx>

#include <iostream>

using ROOT::Experimental::RNTuple;
using ROOT::Experimental::RNTupleInspector;

auto file = TFile::Open("data.rntuple");
auto rntuple = file->Get<RNTuple>("NTupleName");
auto inspector = RNTupleInspector::Create(rntuple).Unwrap();

std::cout << "The compression factor is " << std::fixed << std::setprecision(2)
                                          << inspector->GetCompressionFactor()
                                          << std::endl;
~~~
*/
// clang-format on
class RNTupleInspector {
private:
   std::unique_ptr<ROOT::Experimental::Detail::RPageSource> fPageSource;
   int fCompressionSettings;
   std::uint64_t fCompressedSize;
   std::uint64_t fUncompressedSize;

   RNTupleInspector(std::unique_ptr<ROOT::Experimental::Detail::RPageSource> pageSource)
      : fPageSource(std::move(pageSource)){};

   void CollectSizeData();

public:
   RNTupleInspector(const RNTupleInspector &other) = delete;
   RNTupleInspector &operator=(const RNTupleInspector &other) = delete;
   RNTupleInspector(RNTupleInspector &&other) = delete;
   RNTupleInspector &operator=(RNTupleInspector &&other) = delete;
   ~RNTupleInspector() = default;

   /// Creates a new inspector for a given RNTuple.
   static RResult<std::unique_ptr<RNTupleInspector>>
   Create(std::unique_ptr<ROOT::Experimental::Detail::RPageSource> pageSource);
   static RResult<std::unique_ptr<RNTupleInspector>> Create(RNTuple *sourceNTuple);
   static RResult<std::unique_ptr<RNTupleInspector>> Create(std::string_view ntupleName, std::string_view storage);

   /// Get the name of the RNTuple being inspected.
   std::string GetName();

   /// Get the number of entries in the RNTuple being inspected.
   std::uint64_t GetNEntries();

   /// Get the compression settings of the RNTuple being inspected.
   int GetCompressionSettings();

   /// Get the on-disk, compressed size of the RNTuple being inspected, in bytes.
   /// Does **not** include the size of the header and footer.
   std::uint64_t GetCompressedSize();

   /// Get the total, uncompressed size of the RNTuple being inspected, in bytes.
   /// Does **not** include the size of the header and footer.
   std::uint64_t GetUncompressedSize();

   /// Get the compression factor of the RNTuple being inspected.
   float GetCompressionFactor();
};
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleInspector
