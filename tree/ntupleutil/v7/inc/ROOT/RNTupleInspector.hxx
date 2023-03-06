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

#include <TFile.h>

#include <cstdlib>
#include <memory>
#include <vector>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleInspector
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
   std::unique_ptr<TFile> fSourceFile;
   std::unique_ptr<ROOT::Experimental::Detail::RPageSource> fPageSource;
   std::unique_ptr<ROOT::Experimental::RNTupleDescriptor> fDescriptor;
   int fCompressionSettings;
   std::uint64_t fCompressedSize;
   std::uint64_t fUncompressedSize;

   RNTupleInspector(std::unique_ptr<ROOT::Experimental::Detail::RPageSource> pageSource)
      : fPageSource(std::move(pageSource)){};

   std::vector<DescriptorId_t> GetColumnsForField(DescriptorId_t fieldId);

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

   /// Get the descriptor for the RNTuple being inspected.
   /// Not that this contains a static copy of the descriptor at the time of
   /// creation of the inspector. This means that if the inspected RNTuple changes,
   /// these changes will not be propagated to the RNTupleInspector object!
   RNTupleDescriptor *GetDescriptor();

   /// Get the name of the RNTuple being inspected.
   std::string GetName();

   /// Get the number of entries in the RNTuple being inspected.
   std::uint64_t GetNEntries();

   /// Get the compression settings of the RNTuple being inspected.
   int GetCompressionSettings();

   /// Get the on-disk, compressed size of the RNTuple being inspected, in bytes.
   /// Does **not** include the size of the header and footer.
   std::uint64_t GetOnDiskSize();

   /// Get the total, in-memory size of the RNTuple being inspected, in bytes.
   /// Does **not** include the size of the header and footer.
   std::uint64_t GetInMemorySize();

   /// Get the compression factor of the RNTuple being inspected.
   float GetCompressionFactor();

   /// Get the amount of fields of a given type or class present in the RNTuple.
   int GetFieldTypeFrequency(std::string className);

   /// Get the amount of columns of a given type present in the RNTuple.
   int GetColumnTypeFrequency(EColumnType colType);

   /// Get the on-disk, compressed size of a given column.
   std::uint64_t GetOnDiskColumnSize(DescriptorId_t logicalId);

   /// Get the total, in-memory size of a given column.
   std::uint64_t GetInMemoryColumnSize(DescriptorId_t logicalId);

   /// Get the on-disk, compressed size of a given field.
   std::uint64_t GetOnDiskFieldSize(DescriptorId_t fieldId);
   std::uint64_t GetOnDiskFieldSize(std::string fieldName);

   /// Get the total, in-memory size of a given field.
   std::uint64_t GetInMemoryFieldSize(DescriptorId_t fieldId);
   std::uint64_t GetInMemoryFieldSize(std::string fieldName);

   /// Get the type of a given column.
   EColumnType GetColumnType(DescriptorId_t logicalId);
};
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleInspector
