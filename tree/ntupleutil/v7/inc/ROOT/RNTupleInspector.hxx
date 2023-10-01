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
#include <regex>
#include <vector>

namespace ROOT {
namespace Experimental {

enum class ENTupleInspectorPrintFormat { kTable, kCSV };

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
      std::uint64_t fCompressedSize = 0;
      std::uint32_t fElementSize = 0;
      std::uint64_t fNElements = 0;

   public:
      RColumnInfo(const RColumnDescriptor &colDesc, std::uint64_t onDiskSize, std::uint32_t elemSize,
                  std::uint64_t nElems)
         : fColumnDescriptor(colDesc), fCompressedSize(onDiskSize), fElementSize(elemSize), fNElements(nElems){};
      ~RColumnInfo() = default;

      const RColumnDescriptor &GetDescriptor() const { return fColumnDescriptor; }
      std::uint64_t GetCompressedSize() const { return fCompressedSize; }
      std::uint64_t GetUncompressedSize() const { return fElementSize * fNElements; }
      std::uint64_t GetElementSize() const { return fElementSize; }
      std::uint64_t GetNElements() const { return fNElements; }
      EColumnType GetType() const { return fColumnDescriptor.GetModel().GetType(); }
   };

   /// Holds field-level storage information. Includes storage information of the sub-fields.
   class RFieldTreeInfo {
   private:
      const RFieldDescriptor &fRootFieldDescriptor;
      std::uint64_t fCompressedSize = 0;
      std::uint64_t fUncompressedSize = 0;

   public:
      RFieldTreeInfo(const RFieldDescriptor &fieldDesc, std::uint64_t onDiskSize, std::uint64_t inMemSize)
         : fRootFieldDescriptor(fieldDesc), fCompressedSize(onDiskSize), fUncompressedSize(inMemSize){};
      ~RFieldTreeInfo() = default;

      const RFieldDescriptor &GetDescriptor() const { return fRootFieldDescriptor; }
      std::uint64_t GetCompressedSize() const { return fCompressedSize; }
      std::uint64_t GetUncompressedSize() const { return fUncompressedSize; }
   };

private:
   std::unique_ptr<TFile> fSourceFile;
   std::unique_ptr<Detail::RPageSource> fPageSource;
   std::unique_ptr<RNTupleDescriptor> fDescriptor;
   int fCompressionSettings = -1;
   std::uint64_t fCompressedSize = 0;
   std::uint64_t fUncompressedSize = 0;

   std::map<int, RColumnInfo> fColumnInfo;
   std::map<int, RFieldTreeInfo> fFieldTreeInfo;

   RNTupleInspector(std::unique_ptr<Detail::RPageSource> pageSource);

   /// Gather column-level, as well as RNTuple-level information. The column-level
   /// information will be stored in `fColumnInfo`, and the RNTuple-level information
   /// in `fCompressionSettings`, `fCompressedSize` and `fUncompressedSize`.
   ///
   /// This method is called when the `RNTupleInspector` is initially created. This means that anything unexpected about
   /// the RNTuple itself (e.g. inconsistent compression settings across clusters) will be detected here. Therefore, any
   /// related exceptions will be thrown on creation of the inspector.
   void CollectColumnInfo();

   /// Recursively gather field-level information and store it in `fFieldTreeInfo`.
   ///
   /// This method is called when the `RNTupleInspector` is initially created.
   RFieldTreeInfo CollectFieldTreeInfo(DescriptorId_t fieldId);

   /// Get the IDs of the columns that make up the given field, including its sub-fields.
   std::vector<DescriptorId_t> GetColumnsByFieldId(DescriptorId_t fieldId) const;

public:
   RNTupleInspector(const RNTupleInspector &other) = delete;
   RNTupleInspector &operator=(const RNTupleInspector &other) = delete;
   RNTupleInspector(RNTupleInspector &&other) = delete;
   RNTupleInspector &operator=(RNTupleInspector &&other) = delete;
   ~RNTupleInspector() = default;

   /// Create a new inspector for a given RNTuple. When this factory method is called, all required static information
   /// is collected from the RNTuple's fields and underlying columns are collected at ones. This means that when any
   /// inconsistencies are encountered (e.g. inconsistent compression across clusters), it will throw an error here.
   static std::unique_ptr<RNTupleInspector> Create(std::unique_ptr<Detail::RPageSource> pageSource);
   static std::unique_ptr<RNTupleInspector> Create(RNTuple *sourceNTuple);
   static std::unique_ptr<RNTupleInspector> Create(std::string_view ntupleName, std::string_view storage);

   /// Get the descriptor for the RNTuple being inspected.
   RNTupleDescriptor *GetDescriptor() const { return fDescriptor.get(); }

   /// Get the compression settings of the RNTuple being inspected according to the format described in Compression.h.
   /// Here, we assume that the compression settings are consistent across all clusters and columns. If this is not the
   /// case, an exception will be thrown upon `RNTupleInspector::Create`.
   int GetCompressionSettings() const { return fCompressionSettings; }

   /// Get a description of compression settings of the RNTuple being inspected. Here, we assume that the compression
   /// settings are consistent across all clusters and columns. If this is not the case, an exception will be thrown
   /// upon `RNTupleInspector::Create`.
   std::string GetCompressionSettingsAsString() const;

   /// Get the compressed, on-disk size of the RNTuple being inspected, in bytes.
   /// Does **not** include the size of the header and footer.
   std::uint64_t GetCompressedSize() const { return fCompressedSize; }

   /// Get the uncompressed total size of the RNTuple being inspected, in bytes.
   /// Does **not** include the size of the header and footer.
   std::uint64_t GetUncompressedSize() const { return fUncompressedSize; }

   /// Get the compression factor of the RNTuple being inspected.
   float GetCompressionFactor() const { return (float)fUncompressedSize / (float)fCompressedSize; }

   const RColumnInfo &GetColumnInfo(DescriptorId_t physicalColumnId) const;

   /// Get the number of columns of a given type present in the RNTuple.
   size_t GetColumnCountByType(EColumnType colType) const;

   /// Get the IDs of all columns with the given type.
   const std::vector<DescriptorId_t> GetColumnsByType(EColumnType);

   /// Print the per-column type information, either as a table or in CSV format. The output includes the column type,
   /// its count, the total number of elements, the compressed size and the uncompressed size.
   ///
   /// **Example: printing the column type information of an RNTuple as a table**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleInspector.hxx>
   /// using ROOT::Experimental::RNTupleInspector;
   /// using ROOT::Experimental::ENTupleInspectorPrintFormat;
   ///
   /// auto inspector = RNTupleInspector::Create("myNTuple", "some/file.root");
   /// inspector->PrintColumnTypeInfo();
   /// ~~~
   /// Ouput:
   /// ~~~
   ///  column type    | count   | # elements      | compressed bytes  | uncompressed bytes
   /// ----------------|---------|-----------------|-------------------|--------------------
   ///    SplitIndex64 |       2 |             150 |                72 |               1200
   ///     SplitReal32 |       4 |             300 |               189 |               1200
   ///     SplitUInt32 |       3 |             225 |               123 |                900
   /// ~~~
   ///
   /// **Example: printing the column type information of an RNTuple in CSV format**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleInspector.hxx>
   /// using ROOT::Experimental::RNTupleInspector;
   /// using ROOT::Experimental::ENTupleInspectorPrintFormat;
   ///
   /// auto inspector = RNTupleInspector::Create("myNTuple", "some/file.root");
   /// inspector->PrintColumnTypeInfo();
   /// ~~~
   /// Ouput:
   /// ~~~
   /// columnType,count,nElements,compressedSize,uncompressedSize
   /// SplitIndex64,2,150,72,1200
   /// SplitReal32,4,300,189,1200
   /// SplitUInt32,3,225,123,900
   /// ~~~
   void PrintColumnTypeInfo(ENTupleInspectorPrintFormat format = ENTupleInspectorPrintFormat::kTable,
                            std::ostream &output = std::cout);

   const RFieldTreeInfo &GetFieldTreeInfo(DescriptorId_t fieldId) const;
   const RFieldTreeInfo &GetFieldTreeInfo(std::string_view fieldName) const;

   /// Get the number of fields of a given type or class present in the RNTuple. The type name may contain regular
   /// expression patterns in order to be able to group multiple kinds of types or classes.
   size_t GetFieldCountByType(const std::regex &typeNamePattern, bool searchInSubFields = true) const;
   size_t GetFieldCountByType(std::string_view typeNamePattern, bool searchInSubFields = true) const
   {
      return GetFieldCountByType(std::regex{std::string(typeNamePattern)}, searchInSubFields);
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
