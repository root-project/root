/// \file ROOT/RPageStorage.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-07-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageStorageRoot
#define ROOT7_RPageStorageRoot

#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RColumnModel.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <TDirectory.h>
#include <TFile.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

namespace Internal {

struct RFieldHeader {
   std::int32_t fVersion = 0;
   std::string fName;
   std::string fType;
   std::string fParentName;
};

struct RColumnHeader {
   std::int32_t fVersion = 0;
   std::string fName;
   EColumnType fType;
   bool fIsSorted;
   std::string fOffsetColumn;
};

struct RNTupleHeader {
   std::int32_t fVersion = 0;
   std::string fModelUuid;
   std::vector<RFieldHeader> fFields;
   std::vector<RColumnHeader> fColumns;
};

struct RNTupleFooter {
   std::int32_t fVersion = 0;
   std::int32_t fNClusters = 0;
   NTupleSize_t fNEntries = 0;
   std::vector<NTupleSize_t> fNElementsPerColumn;
};

struct RPageInfo {
   std::vector<NTupleSize_t> fRangeStarts;
};

struct RClusterFooter {
   std::int32_t fVersion = 0;
   NTupleSize_t fEntryRangeStart = 0;
   NTupleSize_t fNEntries = 0;
   std::vector<RPageInfo> fPagesPerColumn;
};

struct RNTupleBlob {
   RNTupleBlob() {}
   RNTupleBlob(int size, unsigned char *content) : fSize(size), fContent(content) {}
   RNTupleBlob(const RNTupleBlob &other) = delete;
   RNTupleBlob &operator =(const RNTupleBlob &other) = delete;
   ~RNTupleBlob() = default;

   std::int32_t fVersion = 0;
   int fSize = 0;
   unsigned char* fContent = nullptr; //[fSize]
};

} // namespace Internal


namespace Detail {

class RPagePool;

/**
 * Maps the ntuple meta-data to and from TFile
 */
class RMapper {
public:
   static constexpr const char* kKeySeparator = "_";
   static constexpr const char* kKeyNTupleHeader = "RFH";
   static constexpr const char* kKeyNTupleFooter = "RFF";
   static constexpr const char* kKeyClusterFooter = "RFCF";
   static constexpr const char* kKeyPagePayload = "RFP";

   struct RColumnIndex {
      NTupleSize_t fNElements = 0;
      std::vector<NTupleSize_t> fRangeStarts;
      std::vector<NTupleSize_t> fClusterId;
      std::vector<NTupleSize_t> fPageInCluster;
      std::vector<NTupleSize_t> fSelfClusterOffset;
      std::vector<NTupleSize_t> fPointeeClusterOffset;
   };

   struct RFieldDescriptor {
      RFieldDescriptor(const std::string &f, const std::string &t) : fFieldName(f), fTypeName(t) {}
      std::string fFieldName;
      std::string fTypeName;
   };

   NTupleSize_t fNEntries = 0;
   std::unordered_map<std::int32_t, std::unique_ptr<RColumnModel>> fId2ColumnModel;
   std::unordered_map<std::string, std::int32_t> fColumnName2Id;
   std::unordered_map<std::int32_t, std::int32_t> fColumn2Pointee;
   std::vector<RColumnIndex> fColumnIndex;
   std::vector<RFieldDescriptor> fRootFields;
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSinkRoot
\ingroup NTuple
\brief Storage provider that write ntuple pages into a ROOT TFile
*/
// clang-format on
class RPageSinkRoot : public RPageSink {
public:
   struct RSettings {
      TFile *fFile = nullptr;
      bool fTakeOwnership = false;
   };

private:
   static constexpr std::size_t kDefaultElementsPerPage = 10000;

   std::unique_ptr<RPageAllocatorHeap> fPageAllocator;

   /// Currently, an ntuple is stored as a directory in a TFile
   TDirectory *fDirectory;
   RSettings fSettings;
   /// Updated on CommitPage and written and reset on CommitCluster
   ROOT::Experimental::Internal::RClusterFooter fCurrentCluster;
   ROOT::Experimental::Internal::RNTupleHeader fNTupleHeader;
   ROOT::Experimental::Internal::RNTupleFooter fNTupleFooter;

   RMapper fMapper;
   NTupleSize_t fPrevClusterNEntries = 0;

   /// Field, column, and cluster ids are issued sequentially starting with 0
   DescriptorId_t fLastFieldId = 0;
   DescriptorId_t fLastColumnId = 0;
   DescriptorId_t fLastClusterId = 0;
   RNTupleDescriptorBuilder fDescriptorBuilder;

public:
   RPageSinkRoot(std::string_view ntupleName, RSettings settings);
   RPageSinkRoot(std::string_view ntupleName, std::string_view path);
   virtual ~RPageSinkRoot();

   ColumnHandle_t AddColumn(const RColumn &column) final;
   void Create(RNTupleModel &model) final;
   void CommitPage(ColumnHandle_t columnHandle, const RPage &page) final;
   void CommitCluster(NTupleSize_t nEntries) final;
   void CommitDataset() final;

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements = 0) final;
   void ReleasePage(RPage &page) final;
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageAllocatorKey
\ingroup NTuple
\brief Adopts the memory returned by TKey->ReadObject()
*/
// clang-format on
class RPageAllocatorKey {
public:
   static RPage NewPage(ColumnId_t columnId, void *mem, std::size_t elementSize, std::size_t nElements);
   static void DeletePage(const RPage& page, ROOT::Experimental::Internal::RNTupleBlob *payload);
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSourceRoot
\ingroup NTuple
\brief Storage provider that reads ntuple pages from a ROOT TFile
*/
// clang-format on
class RPageSourceRoot : public RPageSource {
public:
   struct RSettings {
      TFile *fFile = nullptr;
      bool fTakeOwnership = false;
   };

private:
   std::unique_ptr<RPageAllocatorKey> fPageAllocator;
   std::shared_ptr<RPagePool> fPagePool;

   /// Currently, an ntuple is stored as a directory in a TFile
   TDirectory *fDirectory;
   RSettings fSettings;

   RMapper fMapper;
   RNTupleDescriptor fDescriptor;

public:
   RPageSourceRoot(std::string_view ntupleName, RSettings settings);
   RPageSourceRoot(std::string_view ntupleName, std::string_view path);
   virtual ~RPageSourceRoot();

   ColumnHandle_t AddColumn(const RColumn &column) final;
   void Attach() final;
   std::unique_ptr<ROOT::Experimental::RNTupleModel> GenerateModel() final;
   NTupleSize_t GetNEntries() final;
   NTupleSize_t GetNElements(ColumnHandle_t columnHandle) final;
   ColumnId_t GetColumnId(ColumnHandle_t columnHandle) final;
   const RNTupleDescriptor& GetDescriptor() const final { return fDescriptor; }

   RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t index) final;
   void ReleasePage(RPage &page) final;
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
