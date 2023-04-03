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
public:
   class RColumnInfo {
   private:
      const RColumnDescriptor &fColumnDescriptor;
      std::uint64_t fOnDiskSize = 0;
      std::uint64_t fInMemorySize = 0;
      std::uint32_t fElementSize = 0;
      std::uint64_t fNElements = 0;

   public:
      RColumnInfo(const RColumnDescriptor &colDesc, std::uint64_t onDiskSize, std::uint64_t inMemSize,
                  std::uint32_t elemSize, std::uint64_t nElems)
         : fColumnDescriptor(colDesc),
           fOnDiskSize(onDiskSize),
           fInMemorySize(inMemSize),
           fElementSize(elemSize),
           fNElements(nElems){};
      ~RColumnInfo() = default;

      const RColumnDescriptor &GetDescriptor() { return fColumnDescriptor; }
      std::uint64_t GetOnDiskSize() { return fOnDiskSize; }
      std::uint64_t GetInMemorySize() { return fInMemorySize; }
      std::uint64_t GetElementSize() { return fElementSize; }
      std::uint64_t GetNElements() { return fNElements; }
      EColumnType GetType() { return fColumnDescriptor.GetModel().GetType(); }
   };

   class RFieldInfo {
   private:
      const RFieldDescriptor &fFieldDescriptor;
      std::uint64_t fOnDiskSize = 0;
      std::uint64_t fInMemorySize = 0;

   public:
      RFieldInfo(const RFieldDescriptor &fieldDesc, std::uint64_t onDiskSize, std::uint64_t inMemSize)
         : fFieldDescriptor(fieldDesc), fOnDiskSize(onDiskSize), fInMemorySize(inMemSize){};
      ~RFieldInfo() = default;

      const RFieldDescriptor &GetDescriptor() { return fFieldDescriptor; }
      std::uint64_t GetOnDiskSize() { return fOnDiskSize; }
      std::uint64_t GetInMemorySize() { return fInMemorySize; }
   };

private:
   std::unique_ptr<TFile> fSourceFile;
   std::unique_ptr<Detail::RPageSource> fPageSource;
   std::unique_ptr<RNTupleDescriptor> fDescriptor;
   int fCompressionSettings = -1;
   std::uint64_t fOnDiskSize = 0;
   std::uint64_t fInMemorySize = 0;

   std::vector<RColumnInfo> fColumnInfo;
   std::vector<RFieldInfo> fFieldInfo;

   RNTupleInspector(std::unique_ptr<Detail::RPageSource> pageSource);

   /// Gather column-level, as well as RNTuple-level information. The column-level
   /// information will be stored in `fColumnInfo`, and the RNTuple-level information
   /// in `fCompressionSettings`, `fOnDiskSize` and `fInMemorySize`.
   ///
   /// This method is called when the `RNTupleInspector` is initially created.
   void CollectColumnInfo();

   /// Gather field-level information and store it in `fFieldInfo`.
   ///
   /// Contrary to `CollectColumnInfo`, this method is only called (and thus, `fFieldInfo`
   /// is only populated) when a field-related inspector method is called.
   void CollectFieldInfo();

   /// Get the IDs of the columns that make up the given field, including its sub-fields.
   std::vector<DescriptorId_t> GetColumnsForFieldTree(DescriptorId_t fieldId);

public:
   RNTupleInspector(const RNTupleInspector &other) = delete;
   RNTupleInspector &operator=(const RNTupleInspector &other) = delete;
   RNTupleInspector(RNTupleInspector &&other) = delete;
   RNTupleInspector &operator=(RNTupleInspector &&other) = delete;
   ~RNTupleInspector() = default;

   /// Create a new inspector for a given RNTuple.
   static std::unique_ptr<RNTupleInspector> Create(std::unique_ptr<Detail::RPageSource> pageSource);
   static std::unique_ptr<RNTupleInspector> Create(RNTuple *sourceNTuple);
   static std::unique_ptr<RNTupleInspector> Create(std::string_view ntupleName, std::string_view storage);

   /// Get the descriptor for the RNTuple being inspected.
   RNTupleDescriptor *GetDescriptor() { return fDescriptor.get(); }

   /// Get the compression settings of the RNTuple being inspected.
   int GetCompressionSettings() { return fCompressionSettings; }

   /// Get the on-disk, compressed size of the RNTuple being inspected, in bytes.
   /// Does **not** include the size of the header and footer.
   std::uint64_t GetOnDiskSize() { return fOnDiskSize; }

   /// Get the total, in-memory size of the RNTuple being inspected, in bytes.
   /// Does **not** include the size of the header and footer.
   std::uint64_t GetInMemorySize() { return fInMemorySize; }

   /// Get the compression factor of the RNTuple being inspected.
   float GetCompressionFactor() { return (float)fInMemorySize / (float)fOnDiskSize; }

   const RColumnInfo &GetColumnInfo(DescriptorId_t physicalColumnId);

   const RFieldInfo &GetFieldInfo(DescriptorId_t fieldId);
   const RFieldInfo &GetFieldInfo(std::string_view fieldName);

   /// Get the number of fields of a given type or class present in the RNTuple.
   /// TODO: Add regex support.
   int GetFieldTypeCount(std::string_view typeName, bool includeSubFields = true);

   /// Get the number of columns of a given type present in the RNTuple.
   int GetColumnTypeCount(EColumnType colType);
};
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleInspector
