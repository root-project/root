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
#include <regex>

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
   /// Holds column-level storage information.
   class RColumnInfo {
   private:
      const RColumnDescriptor &fColumnDescriptor;
      std::uint64_t fOnDiskSize = 0;
      std::uint32_t fElementSize = 0;
      std::uint64_t fNElements = 0;

   public:
      RColumnInfo(const RColumnDescriptor &colDesc, std::uint64_t onDiskSize, std::uint32_t elemSize,
                  std::uint64_t nElems)
         : fColumnDescriptor(colDesc), fOnDiskSize(onDiskSize), fElementSize(elemSize), fNElements(nElems){};
      ~RColumnInfo() = default;

      const RColumnDescriptor &GetDescriptor() const { return fColumnDescriptor; }
      std::uint64_t GetOnDiskSize() const { return fOnDiskSize; }
      std::uint64_t GetInMemorySize() const { return fElementSize * fNElements; }
      std::uint64_t GetElementSize() const { return fElementSize; }
      std::uint64_t GetNElements() const { return fNElements; }
      EColumnType GetType() const { return fColumnDescriptor.GetModel().GetType(); }
   };

   /// Holds field-level storage information. Includes storage information of the sub-fields.
   class RFieldTreeInfo {
   private:
      const RFieldDescriptor &fRootFieldDescriptor;
      std::uint64_t fOnDiskSize = 0;
      std::uint64_t fInMemorySize = 0;

   public:
      RFieldTreeInfo(const RFieldDescriptor &fieldDesc, std::uint64_t onDiskSize, std::uint64_t inMemSize)
         : fRootFieldDescriptor(fieldDesc), fOnDiskSize(onDiskSize), fInMemorySize(inMemSize){};
      ~RFieldTreeInfo() = default;

      const RFieldDescriptor &GetDescriptor() const { return fRootFieldDescriptor; }
      std::uint64_t GetOnDiskSize() const { return fOnDiskSize; }
      std::uint64_t GetInMemorySize() const { return fInMemorySize; }
   };

private:
   std::unique_ptr<TFile> fSourceFile;
   std::unique_ptr<Detail::RPageSource> fPageSource;
   std::unique_ptr<RNTupleDescriptor> fDescriptor;
   int fCompressionSettings = -1;
   std::uint64_t fOnDiskSize = 0;
   std::uint64_t fInMemorySize = 0;

   std::map<int, RColumnInfo> fColumnInfo;
   std::map<int, RFieldTreeInfo> fFieldTreeInfo;

   RNTupleInspector(std::unique_ptr<Detail::RPageSource> pageSource);

   /// Gather column-level, as well as RNTuple-level information. The column-level
   /// information will be stored in `fColumnInfo`, and the RNTuple-level information
   /// in `fCompressionSettings`, `fOnDiskSize` and `fInMemorySize`.
   ///
   /// This method is called when the `RNTupleInspector` is initially created.
   void CollectColumnInfo();

   /// Recursively gather field-level information and store it in `fFieldTreeInfo`.
   ///
   /// This method is called when the `RNTupleInspector` is initially created.
   RFieldTreeInfo CollectFieldTreeInfo(DescriptorId_t fieldId);

   /// Get the IDs of the columns that make up the given field, including its sub-fields.
   std::vector<DescriptorId_t> GetColumnsForFieldTree(DescriptorId_t fieldId) const;

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
   RNTupleDescriptor *GetDescriptor() const { return fDescriptor.get(); }

   /// Get the compression settings of the RNTuple being inspected.
   int GetCompressionSettings() const { return fCompressionSettings; }

   /// Get the on-disk, compressed size of the RNTuple being inspected, in bytes.
   /// Does **not** include the size of the header and footer.
   std::uint64_t GetOnDiskSize() const { return fOnDiskSize; }

   /// Get the total, in-memory size of the RNTuple being inspected, in bytes.
   /// Does **not** include the size of the header and footer.
   std::uint64_t GetInMemorySize() const { return fInMemorySize; }

   /// Get the compression factor of the RNTuple being inspected.
   float GetCompressionFactor() const { return (float)fInMemorySize / (float)fOnDiskSize; }

   const RColumnInfo &GetColumnInfo(DescriptorId_t physicalColumnId) const;

   /// Get the number of columns of a given type present in the RNTuple.
   size_t GetColumnTypeCount(EColumnType colType) const;

   const RFieldTreeInfo &GetFieldTreeInfo(DescriptorId_t fieldId) const;
   const RFieldTreeInfo &GetFieldTreeInfo(std::string_view fieldName) const;

   /// Get the number of fields of a given type or class present in the RNTuple. The type name may contain regular
   /// expression patterns in order to be able to group multiple kinds of types or classes.
   size_t GetFieldTypeCount(const std::regex &typeNamePattern, bool searchInSubFields = true) const;
   size_t GetFieldTypeCount(std::string_view typeNamePattern, bool searchInSubFields = true) const
   {
      return GetFieldTypeCount(std::regex{std::string(typeNamePattern)}, searchInSubFields);
   }

   /// Get the IDs of (sub-)fields whose name matches the given string. Because field names are unique by design,
   /// providing a single field name will return a vector containing just the ID of that field. However, regular
   /// expression patterns are supported in order to get the IDs of all fields whose name follow a certain structure.
   const std::vector<DescriptorId_t>
   GetFieldsByName(const std::regex &fieldNamePattern, bool searchInSubFields = true) const;
   const std::vector<DescriptorId_t> GetFieldsByName(std::string_view fieldNamePattern, bool searchInSubFields = true)
   {
      return GetFieldsByName(std::regex{std::string(fieldNamePattern)}, searchInSubFields);
   }
};
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleInspector
