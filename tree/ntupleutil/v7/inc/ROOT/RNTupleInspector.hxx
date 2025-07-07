/// \file ROOT/RNTupleInspector.hxx
/// \ingroup NTuple ROOT7
/// \author Florine de Geus <florine.de.geus@cern.ch>
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
#include <ROOT/RNTupleDescriptor.hxx>

#include <TFile.h>
#include <TH1D.h>
#include <THStack.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <regex>
#include <vector>

namespace ROOT {
class RNTuple;

namespace Internal {
class RPageSource;
} // namespace Internal

namespace Experimental {

enum class ENTupleInspectorPrintFormat { kTable, kCSV };
enum class ENTupleInspectorHist { kCount, kNElems, kCompressedSize, kUncompressedSize };

// clang-format off
/**
\class ROOT::Experimental::RNTupleInspector
\ingroup NTuple
\brief Inspect on-disk and storage-related information of an RNTuple.

The RNTupleInspector can be used for studying an RNTuple in terms of its storage efficiency. It provides information on
the level of the RNTuple itself, on the (sub)field level and on the column level.

Example usage:

~~~ {.cpp}
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleInspector.hxx>

#include <iostream>

using ROOT::Experimental::RNTupleInspector;

auto file = TFile::Open("data.rntuple");
auto rntuple = std::unique_ptr<ROOT::RNTuple>(file->Get<RNTuple>("NTupleName"));
auto inspector = RNTupleInspector::Create(*rntuple);

std::cout << "The compression factor is " << inspector->GetCompressionFactor()
          << " using compression settings " << inspector->GetCompressionSettingsAsString()
          << std::endl;
~~~
*/
// clang-format on
class RNTupleInspector {
public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Provides column-level storage information.
   ///
   /// The RColumnInspector class provides storage information for an individual column. This information is partly
   /// collected during the construction of the RNTupleInspector object, and can partly be accessed using the
   /// RColumnInspector that belongs to this field.
   class RColumnInspector {
   private:
      const ROOT::RColumnDescriptor &fColumnDescriptor;
      const std::vector<std::uint64_t> fCompressedPageSizes = {};
      std::uint32_t fElementSize = 0;
      std::uint64_t fNElements = 0;

   public:
      RColumnInspector(const ROOT::RColumnDescriptor &colDesc, const std::vector<std::uint64_t> &compressedPageSizes,
                       std::uint32_t elemSize, std::uint64_t nElems)
         : fColumnDescriptor(colDesc),
           fCompressedPageSizes(compressedPageSizes),
           fElementSize(elemSize),
           fNElements(nElems) {};
      ~RColumnInspector() = default;

      const ROOT::RColumnDescriptor &GetDescriptor() const { return fColumnDescriptor; }
      const std::vector<std::uint64_t> &GetCompressedPageSizes() const { return fCompressedPageSizes; }
      std::uint64_t GetNPages() const { return fCompressedPageSizes.size(); }
      std::uint64_t GetCompressedSize() const
      {
         return std::accumulate(fCompressedPageSizes.begin(), fCompressedPageSizes.end(), static_cast<std::uint64_t>(0));
      }
      std::uint64_t GetUncompressedSize() const { return fElementSize * fNElements; }
      std::uint64_t GetElementSize() const { return fElementSize; }
      std::uint64_t GetNElements() const { return fNElements; }
      ROOT::ENTupleColumnType GetType() const { return fColumnDescriptor.GetType(); }
   };

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Provides field-level storage information.
   ///
   /// The RFieldTreeInspector class provides storage information for a field **and** its subfields. This information is
   /// partly collected during the construction of the RNTupleInspector object, and can partly be accessed using
   /// the RFieldDescriptor that belongs to this field.
   class RFieldTreeInspector {
   private:
      const ROOT::RFieldDescriptor &fRootFieldDescriptor;
      std::uint64_t fCompressedSize = 0;
      std::uint64_t fUncompressedSize = 0;

   public:
      RFieldTreeInspector(const ROOT::RFieldDescriptor &fieldDesc, std::uint64_t onDiskSize, std::uint64_t inMemSize)
         : fRootFieldDescriptor(fieldDesc), fCompressedSize(onDiskSize), fUncompressedSize(inMemSize) {};
      ~RFieldTreeInspector() = default;

      const ROOT::RFieldDescriptor &GetDescriptor() const { return fRootFieldDescriptor; }
      std::uint64_t GetCompressedSize() const { return fCompressedSize; }
      std::uint64_t GetUncompressedSize() const { return fUncompressedSize; }
   };

private:
   std::unique_ptr<ROOT::Internal::RPageSource> fPageSource;
   ROOT::RNTupleDescriptor fDescriptor;
   std::optional<std::uint32_t> fCompressionSettings; ///< The compression settings are unknown for an empty ntuple
   std::uint64_t fCompressedSize = 0;
   std::uint64_t fUncompressedSize = 0;

   std::unordered_map<int, RColumnInspector> fColumnInfo;
   std::unordered_map<int, RFieldTreeInspector> fFieldTreeInfo;

   RNTupleInspector(std::unique_ptr<ROOT::Internal::RPageSource> pageSource);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Gather column-level and RNTuple-level information.
   ///
   /// \note This method is called when the RNTupleInspector is initially created. This means that anything unexpected
   /// about the RNTuple itself (e.g. inconsistent compression settings across clusters) will be detected here.
   /// Therefore, any related exceptions will be thrown on creation of the inspector.
   void CollectColumnInfo();

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Recursively gather field-level information.
   ///
   /// \param[in] fieldId The ID of the field from which to start the recursive traversal. Typically this is the "zero
   /// ID", i.e. the logical parent of all top-level fields.
   ///
   /// \return The RFieldTreeInspector for the provided field ID.
   ///
   /// This method is called when the RNTupleInspector is initially created.
   RFieldTreeInspector CollectFieldTreeInfo(ROOT::DescriptorId_t fieldId);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the columns that make up the given field, including its subfields.
   ///
   /// \param [in] fieldId The ID of the field for which to collect the columns.
   ///
   /// \return A vector containing the IDs of all columns for the provided field ID.
   std::vector<ROOT::DescriptorId_t> GetColumnsByFieldId(ROOT::DescriptorId_t fieldId) const;

public:
   RNTupleInspector(const RNTupleInspector &other) = delete;
   RNTupleInspector &operator=(const RNTupleInspector &other) = delete;
   RNTupleInspector(RNTupleInspector &&other) = delete;
   RNTupleInspector &operator=(RNTupleInspector &&other) = delete;
   ~RNTupleInspector();

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new RNTupleInspector.
   ///
   /// \param[in] sourceNTuple A pointer to the RNTuple to be inspected.
   ///
   /// \return A pointer to the newly created RNTupleInspector.
   ///
   /// \note When this factory method is called, all required static information is collected from the RNTuple's fields
   /// and underlying columns are collected at ones. This means that when any inconsistencies are encountered (e.g.
   /// inconsistent compression across clusters), it will throw an error here.
   static std::unique_ptr<RNTupleInspector> Create(const RNTuple &sourceNTuple);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new RNTupleInspector.
   ///
   /// \param[in] ntupleName The name of the RNTuple to be inspected.
   /// \param[in] storage The path or URI to the RNTuple to be inspected.
   ///
   /// \see Create(RNTuple *sourceNTuple)
   static std::unique_ptr<RNTupleInspector> Create(std::string_view ntupleName, std::string_view storage);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the descriptor for the RNTuple being inspected.
   ///
   /// \return A static copy of the ROOT::RNTupleDescriptor belonging to the inspected RNTuple.
   const ROOT::RNTupleDescriptor &GetDescriptor() const { return fDescriptor; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the compression settings of the RNTuple being inspected.
   ///
   /// \return The integer representation (\f$algorithm * 10 + level\f$, where \f$algorithm\f$ follows
   /// ROOT::RCompressionSetting::ELevel::EValues) of the compression settings used for the inspected RNTuple.
   /// Empty for an empty ntuple.
   ///
   /// \note Here, we assume that the compression settings are consistent across all clusters and columns. If this is
   /// not the case, an exception will be thrown when RNTupleInspector::Create is called.
   std::optional<std::uint32_t> GetCompressionSettings() const { return fCompressionSettings; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a string describing compression settings of the RNTuple being inspected.
   ///
   /// \return A string describing the compression used for the inspected RNTuple. The format of the string is
   /// `"A (level L)"`, where `A` is the name of the compression algorithm and `L` the compression level.
   ///
   /// \note Here, we assume that the compression settings are consistent across all clusters and columns. If this is
   /// not the case, an exception will be thrown when RNTupleInspector::Create is called.
   std::string GetCompressionSettingsAsString() const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the compressed, on-disk size of the RNTuple being inspected.
   ///
   /// \return The compressed size of the inspected RNTuple, in bytes, excluding the size of the header and footer.
   std::uint64_t GetCompressedSize() const { return fCompressedSize; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the uncompressed total size of the RNTuple being inspected.
   ///
   /// \return The uncompressed size of the inspected RNTuple, in bytes, excluding the size of the header and footer.
   std::uint64_t GetUncompressedSize() const { return fUncompressedSize; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the compression factor of the RNTuple being inspected.
   ///
   /// \return The compression factor of the inspected RNTuple.
   ///
   /// The compression factor shows how well the data present in the RNTuple is compressed by the compression settings
   /// that were used. The compression factor is calculated as \f$size_{uncompressed} / size_{compressed}\f$.
   float GetCompressionFactor() const { return (float)fUncompressedSize / (float)fCompressedSize; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get storage information for a given column.
   ///
   /// \param[in] physicalColumnId The physical ID of the column for which to get the information.
   ///
   /// \return The storage information for the provided column.
   const RColumnInspector &GetColumnInspector(ROOT::DescriptorId_t physicalColumnId) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the number of columns of a given type present in the RNTuple.
   ///
   /// \param[in] colType The column type to count, as defined by ROOT::ENTupleColumnType.
   ///
   /// \return The number of columns present in the inspected RNTuple of the provided type.
   size_t GetColumnCountByType(ROOT::ENTupleColumnType colType) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the IDs of all columns with the given type.
   ///
   /// \param[in] colType The column type to collect, as defined by ROOT::ENTupleColumnType.
   ///
   /// \return A vector containing the physical IDs of columns of the provided type.
   const std::vector<ROOT::DescriptorId_t> GetColumnsByType(ROOT::ENTupleColumnType colType);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all column types present in the RNTuple being inspected.
   ///
   /// \return A vector containing all column types present in the RNTuple.
   const std::vector<ROOT::ENTupleColumnType> GetColumnTypes();

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Print storage information per column type.
   ///
   /// \param[in] format Whether to print the information as a (markdown-parseable) table or in CSV format.
   /// \param[in] output Where to write the output to. Default is `stdout`.
   ///
   /// The output includes for each column type its count, the total number of elements, the compressed size and the
   /// uncompressed size.
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
   /// Output:
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
   /// Output:
   /// ~~~
   /// columnType,count,nElements,compressedSize,uncompressedSize
   /// SplitIndex64,2,150,72,1200
   /// SplitReal32,4,300,189,1200
   /// SplitUInt32,3,225,123,900
   /// ~~~
   void PrintColumnTypeInfo(ENTupleInspectorPrintFormat format = ENTupleInspectorPrintFormat::kTable,
                            std::ostream &output = std::cout);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a histogram showing information for each column type present,
   ///
   /// \param[in] histKind Which type of information should be returned.
   /// \param[in] histName The name of the histogram. An empty string means a default name will be used.
   /// \param[in] histTitle The title of the histogram. An empty string means a default title will be used.
   ///
   /// \return A pointer to a `TH1D` containing the specified kind of information.
   ///
   /// Get a histogram showing the count, number of elements, size on disk, or size in memory for each column
   /// type present in the inspected RNTuple.
   std::unique_ptr<TH1D> GetColumnTypeInfoAsHist(ENTupleInspectorHist histKind, std::string_view histName = "",
                                                 std::string_view histTitle = "");

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a histogram containing the size distribution of the compressed pages for an individual column.
   ///
   /// \param[in] physicalColumnId The physical ID of the column for which to get the page size distribution.
   /// \param[in] histName The name of the histogram. An empty string means a default name will be used.
   /// \param[in] histTitle The title of the histogram. An empty string means a default title will be used.
   /// \param[in] nBins The desired number of histogram bins.
   ///
   /// \return A pointer to a `TH1D` containing the page size distribution.
   ///
   /// The x-axis will range from the smallest page size, to the largest (inclusive).
   std::unique_ptr<TH1D> GetPageSizeDistribution(ROOT::DescriptorId_t physicalColumnId, std::string histName = "",
                                                 std::string histTitle = "", size_t nBins = 64);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a histogram containing the size distribution of the compressed pages for all columns of a given type.
   ///
   /// \param[in] colType The column type for which to get the size distribution, as defined by ROOT::ENTupleColumnType.
   /// \param[in] histName The name of the histogram. An empty string means a default name will be used.
   /// \param[in] histTitle The title of the histogram. An empty string means a default title will be used.
   /// \param[in] nBins The desired number of histogram bins.
   ///
   /// \return A pointer to a `TH1D` containing the page size distribution.
   ///
   /// The x-axis will range from the smallest page size, to the largest (inclusive).
   std::unique_ptr<TH1D> GetPageSizeDistribution(ROOT::ENTupleColumnType colType, std::string histName = "",
                                                 std::string histTitle = "", size_t nBins = 64);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a histogram containing the size distribution of the compressed pages for a collection columns.
   ///
   /// \param[in] colIds The physical IDs of the columns for which to get the page size distribution.
   /// \param[in] histName The name of the histogram. An empty string means a default name will be used.
   /// \param[in] histTitle The title of the histogram. An empty string means a default title will be used.
   /// \param[in] nBins The desired number of histogram bins.
   ///
   /// \return A pointer to a `TH1D` containing the (cumulative) page size distribution.
   ///
   /// The x-axis will range from the smallest page size, to the largest (inclusive).
   std::unique_ptr<TH1D> GetPageSizeDistribution(std::initializer_list<ROOT::DescriptorId_t> colIds,
                                                 std::string histName = "", std::string histTitle = "",
                                                 size_t nBins = 64);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a histogram containing the size distribution of the compressed pages for all columns of a given list
   /// of types.
   ///
   /// \param[in] colTypes The column types for which to get the size distribution, as defined by
   /// ROOT::ENTupleColumnType. The default is an empty vector, which indicates that the distribution
   /// for *all* physical columns will be returned.
   /// \param[in] histName The name of the histogram. An empty string means a default name will be used. The name of
   /// each histogram inside the `THStack` will be `histName + colType`.
   /// \param[in] histTitle The title of the histogram. An empty string means a default title will be used.
   /// \param[in] nBins The desired number of histogram bins.
   ///
   /// \return A pointer to a `THStack` with one histogram for each column type.
   ///
   /// The x-axis will range from the smallest page size, to the largest (inclusive).
   ///
   /// **Example: Drawing a non-stacked page size distribution with a legend**
   /// ~~~ {.cpp}
   /// auto canvas = std::make_unique<TCanvas>();
   /// auto inspector = RNTupleInspector::Create("myNTuple", "ntuple.root");
   ///
   /// // We want to show the page size distributions of columns with type `kSplitReal32` and `kSplitReal64`.
   /// auto hist = inspector->GetPageSizeDistribution(
   ///     {ROOT::ENTupleColumnType::kSplitReal32, ROOT::ENTupleColumnType::kSplitReal64});
   /// // The "PLC" option automatically sets the line color for each histogram in the `THStack`.
   /// // The "NOSTACK" option will draw the histograms on top of each other instead of stacked.
   /// hist->DrawClone("PLC NOSTACK");
   /// canvas->BuildLegend(0.7, 0.8, 0.89, 0.89);
   /// canvas->DrawClone();
   /// ~~~
   std::unique_ptr<THStack> GetPageSizeDistribution(std::initializer_list<ROOT::ENTupleColumnType> colTypes = {},
                                                    std::string histName = "", std::string histTitle = "",
                                                    size_t nBins = 64);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get storage information for a given (sub)field by ID.
   ///
   /// \param[in] fieldId The ID of the (sub)field for which to get the information.
   ///
   /// \return The storage information inspector for the provided (sub)field tree.
   const RFieldTreeInspector &GetFieldTreeInspector(ROOT::DescriptorId_t fieldId) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a storage information inspector for a given (sub)field by name, including its subfields.
   ///
   /// \param[in] fieldName The name of the (sub)field for which to get the information.
   ///
   /// \return The storage information inspector for the provided (sub)field tree.
   const RFieldTreeInspector &GetFieldTreeInspector(std::string_view fieldName) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the number of fields of a given type or class present in the RNTuple.
   ///
   /// \param[in] typeNamePattern The type or class name to count. May contain regular expression patterns for grouping
   /// multiple kinds of types or classes.
   /// \param[in] searchInSubfields If set to `false`, only top-level fields will be considered.
   ///
   /// \return The number of fields that matches the provided type.
   size_t GetFieldCountByType(const std::regex &typeNamePattern, bool searchInSubfields = true) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the number of fields of a given type or class present in the RNTuple.
   ///
   /// \see GetFieldCountByType(const std::regex &typeNamePattern, bool searchInSubfields) const
   size_t GetFieldCountByType(std::string_view typeNamePattern, bool searchInSubfields = true) const
   {
      return GetFieldCountByType(std::regex{std::string(typeNamePattern)}, searchInSubfields);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the IDs of (sub-)fields whose name matches the given string.
   ///
   /// \param[in] fieldNamePattern The name of the field name to get. Because field names are unique by design,
   /// providing a single field name will return a vector containing just the ID of that field. However, regular
   /// expression patterns are supported in order to get the IDs of all fields whose name follow a certain structure.
   /// \param[in] searchInSubfields If set to `false`, only top-level fields will be considered.
   ///
   /// \return A vector containing the IDs of fields that match the provided name.
   const std::vector<ROOT::DescriptorId_t>
   GetFieldsByName(const std::regex &fieldNamePattern, bool searchInSubfields = true) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the IDs of (sub-)fields whose name matches the given string.
   ///
   /// \see GetFieldsByName(const std::regex &fieldNamePattern, bool searchInSubfields) const
   const std::vector<ROOT::DescriptorId_t>
   GetFieldsByName(std::string_view fieldNamePattern, bool searchInSubfields = true)
   {
      return GetFieldsByName(std::regex{std::string(fieldNamePattern)}, searchInSubfields);
   }
};
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleInspector
