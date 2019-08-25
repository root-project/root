/// \file RPageStorage.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RField.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/RPageStorageRoot.hxx>
#include <ROOT/RLogger.hxx>

#include <TKey.h>

#include <cstdlib>
#include <iostream>
#include <utility>

namespace {

static constexpr const char* kKeySeparator = "_";
static constexpr const char* kKeyNTupleFooter = "NTPLF";
static constexpr const char* kKeyNTupleHeader = "NTPLH";
static constexpr const char* kKeyPagePayload = "NTPLP";

}

ROOT::Experimental::Detail::RPageSinkRoot::RPageSinkRoot(std::string_view ntupleName, std::string_view path,
   const RNTupleWriteOptions &options)
   : RPageSink(ntupleName, options)
   , fPageAllocator(std::make_unique<RPageAllocatorHeap>())
{
   R__WARNING_HERE("NTuple") << "The RNTuple file format will change. " <<
      "Do not store real data with this version of RNTuple!";
   fFile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "RECREATE"));
   fFile->SetCompressionSettings(fOptions.GetCompression());
}

ROOT::Experimental::Detail::RPageSinkRoot::~RPageSinkRoot()
{
   if (fFile)
      fFile->Close();
}

void ROOT::Experimental::Detail::RPageSinkRoot::DoCreate(const RNTupleModel & /* model */)
{
   fDirectory = fFile->mkdir(fNTupleName.c_str());
   // In TBrowser, use RNTupleBrowser(TDirectory *directory) in order to show the ntuple contents
   fDirectory->SetBit(TDirectoryFile::kCustomBrowse);
   fDirectory->SetTitle("ROOT::Experimental::Detail::RNTupleBrowser");

   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szHeader = descriptor.SerializeHeader(nullptr);
   auto buffer = new unsigned char[szHeader];
   descriptor.SerializeHeader(buffer);
   ROOT::Experimental::Internal::RNTupleBlob blob(szHeader, buffer);
   fDirectory->WriteObject(&blob, kKeyNTupleHeader);
   delete[] buffer;
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkRoot::DoCommitPage(ColumnHandle_t columnHandle, const RPage &page)
{
   unsigned char *buffer = reinterpret_cast<unsigned char *>(page.GetBuffer());
   auto packedBytes = page.GetSize();
   auto element = columnHandle.fColumn->GetElement();
   const auto isMappable = element->IsMappable();

   if (!isMappable) {
      packedBytes = (page.GetNElements() * element->GetBitsOnStorage() + 7) / 8;
      buffer = new unsigned char[packedBytes];
      element->Pack(buffer, page.GetBuffer(), page.GetNElements());
   }

   ROOT::Experimental::Internal::RNTupleBlob pagePayload(packedBytes, buffer);
   std::string keyName = std::string(kKeyPagePayload) +
      std::to_string(fLastClusterId) + kKeySeparator +
      std::to_string(fLastPageIdx);
   fDirectory->WriteObject(&pagePayload, keyName.c_str());

   if (!isMappable) {
      delete[] buffer;
   }

   RClusterDescriptor::RLocator result;
   result.fPosition = fLastPageIdx++;
   result.fBytesOnStorage = packedBytes;
   return result;
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkRoot::DoCommitCluster(ROOT::Experimental::NTupleSize_t /* nEntries */)
{
   fLastPageIdx = 0;
   return RClusterDescriptor::RLocator();
}

void ROOT::Experimental::Detail::RPageSinkRoot::DoCommitDataset()
{
   if (!fDirectory)
      return;

   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szFooter = descriptor.SerializeFooter(nullptr);
   auto buffer = new unsigned char[szFooter];
   descriptor.SerializeFooter(buffer);
   ROOT::Experimental::Internal::RNTupleBlob footerBlob(szFooter, buffer);
   fDirectory->WriteObject(&footerBlob, kKeyNTupleFooter);
   delete[] buffer;
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSinkRoot::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   if (nElements == 0)
      nElements = kDefaultElementsPerPage;
   auto elementSize = columnHandle.fColumn->GetElement()->GetSize();
   return fPageAllocator->NewPage(columnHandle.fId, elementSize, nElements);
}

void ROOT::Experimental::Detail::RPageSinkRoot::ReleasePage(RPage &page)
{
   fPageAllocator->DeletePage(page);
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageAllocatorKey::NewPage(
   ColumnId_t columnId, void *mem, std::size_t elementSize, std::size_t nElements)
{
   RPage newPage(columnId, mem, elementSize * nElements, elementSize);
   newPage.TryGrow(nElements);
   return newPage;
}

void ROOT::Experimental::Detail::RPageAllocatorKey::DeletePage(
   const RPage& page, ROOT::Experimental::Internal::RNTupleBlob *payload)
{
   if (page.IsNull())
      return;
   R__ASSERT(page.GetBuffer() == payload->fContent);
   free(payload->fContent);
   delete payload;
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPageSourceRoot::RPageSourceRoot(std::string_view ntupleName, std::string_view path,
   const RNTupleReadOptions &options)
   : RPageSource(ntupleName, options)
   , fPageAllocator(std::make_unique<RPageAllocatorKey>())
   , fPagePool(std::make_shared<RPagePool>())
{
   fFile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "READ"));
}


ROOT::Experimental::Detail::RPageSourceRoot::~RPageSourceRoot()
{
   if (fFile)
      fFile->Close();
}


ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Detail::RPageSourceRoot::DoAttach()
{
   fDirectory = fFile->GetDirectory(fNTupleName.c_str());
   RNTupleDescriptorBuilder descBuilder;

   auto keyRawNTupleHeader = fDirectory->GetKey(kKeyNTupleHeader);
   auto ntupleRawHeader = keyRawNTupleHeader->ReadObject<ROOT::Experimental::Internal::RNTupleBlob>();
   descBuilder.SetFromHeader(ntupleRawHeader->fContent);
   free(ntupleRawHeader->fContent);
   delete ntupleRawHeader;

   auto keyRawNTupleFooter = fDirectory->GetKey(kKeyNTupleFooter);
   auto ntupleRawFooter = keyRawNTupleFooter->ReadObject<ROOT::Experimental::Internal::RNTupleBlob>();
   descBuilder.AddClustersFromFooter(ntupleRawFooter->fContent);
   free(ntupleRawFooter->fContent);
   delete ntupleRawFooter;

   return descBuilder.GetDescriptor();
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceRoot::PopulatePageFromCluster(
   ColumnHandle_t columnHandle, const RClusterDescriptor &clusterDescriptor, ClusterSize_t::ValueType clusterIndex)
{
   auto columnId = columnHandle.fId;
   auto clusterId = clusterDescriptor.GetId();
   auto pageRange = clusterDescriptor.GetPageRange(columnId);

   // TODO(jblomer): binary search
   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   decltype(clusterIndex) firstInPage = 0;
   for (const auto &pi : pageRange.fPageInfos) {
      if (firstInPage + pi.fNElements > clusterIndex) {
         pageInfo = pi;
         break;
      }
      firstInPage += pi.fNElements;
   }
   R__ASSERT(firstInPage <= clusterIndex);
   R__ASSERT((firstInPage + pageInfo.fNElements) > clusterIndex);

   //printf("Populating page %lu/%lu [%lu] for column %d starting at %lu\n", clusterId, pageInCluster, pageIdx, columnId, firstInPage);

   std::string keyName = std::string(kKeyPagePayload) +
      std::to_string(clusterId) + kKeySeparator +
      std::to_string(pageInfo.fLocator.fPosition);
   auto pageKey = fDirectory->GetKey(keyName.c_str());
   auto pagePayload = pageKey->ReadObject<ROOT::Experimental::Internal::RNTupleBlob>();

   unsigned char *buffer = pagePayload->fContent;
   auto element = columnHandle.fColumn->GetElement();
   auto elementSize = element->GetSize();
   if (!element->IsMappable()) {
      auto pageSize = elementSize * pageInfo.fNElements;
      buffer = reinterpret_cast<unsigned char *>(malloc(pageSize));
      R__ASSERT(buffer != nullptr);
      element->Unpack(buffer, pagePayload->fContent, pageInfo.fNElements);
      free(pagePayload->fContent);
      pagePayload->fContent = buffer;
      pagePayload->fSize = pageSize;
   }

   auto indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   auto newPage = fPageAllocator->NewPage(columnId, pagePayload->fContent, elementSize, pageInfo.fNElements);
   newPage.SetWindow(indexOffset + firstInPage, RPage::RClusterInfo(clusterId, indexOffset));
   fPagePool->RegisterPage(newPage,
      RPageDeleter([](const RPage &page, void *userData)
      {
         RPageAllocatorKey::DeletePage(page, reinterpret_cast<ROOT::Experimental::Internal::RNTupleBlob *>(userData));
      }, pagePayload));
   return newPage;
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceRoot::PopulatePage(
   ColumnHandle_t columnHandle, NTupleSize_t globalIndex)
{
   auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, globalIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   auto clusterId = fDescriptor.FindClusterId(columnId, globalIndex);
   R__ASSERT(clusterId != kInvalidDescriptorId);
   auto clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   auto selfOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   R__ASSERT(selfOffset <= globalIndex);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, globalIndex - selfOffset);
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceRoot::PopulatePage(
   ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex)
{
   auto clusterId = clusterIndex.GetClusterId();
   auto index = clusterIndex.GetIndex();
   auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, clusterIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   R__ASSERT(clusterId != kInvalidDescriptorId);
   auto clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, index);
}

void ROOT::Experimental::Detail::RPageSourceRoot::ReleasePage(RPage &page)
{
   fPagePool->ReturnPage(page);
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSource> ROOT::Experimental::Detail::RPageSourceRoot::Clone() const
{
   return std::make_unique<RPageSourceRoot>(fNTupleName, fFile->GetName(), fOptions);
}
