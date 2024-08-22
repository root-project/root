/// \file RNTupleMerger.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>, Max Orok <maxwellorok@gmail.com>, Alaettin Serhan Mete <amete@anl.gov>,
/// Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2020-07-08 \warning This is part of the ROOT 7 prototype! It will
/// change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleMerger.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RClusterPool.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/TTaskGroup.hxx>
#include <ROOT/RColumnElementBase.hxx>
#include <TROOT.h>
#include <TFileMergeInfo.h>
#include <TError.h>
#include <TFile.h>
#include <TKey.h>

#include <deque>
#include <algorithm>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Internal;

// Entry point for TFileMerger. Internally calls RNTupleMerger::Merge().
Long64_t RNTuple::Merge(TCollection *inputs, TFileMergeInfo *mergeInfo)
// IMPORTANT: this function must not throw, as it is used in exception-unsafe code (TFileMerger).
try {
   // Check the inputs
   if (!inputs || inputs->GetEntries() < 3 || !mergeInfo) {
      Error("RNTuple::Merge", "Invalid inputs.");
      return -1;
   }

   // Parse the input parameters
   TIter itr(inputs);

   // First entry is the RNTuple name
   std::string ntupleName = std::string(itr()->GetName());

   // Second entry is the output file
   TObject *secondArg = itr();
   TFile *outFile = dynamic_cast<TFile *>(secondArg);
   if (!outFile) {
      Error("RNTuple::Merge", "Second input parameter should be a TFile, but it's a %s.", secondArg->ClassName());
      return -1;
   }

   // Check if the output file already has a key with that name
   TKey *outKey = outFile->FindKey(ntupleName.c_str());
   RNTuple *outNTuple = nullptr;
   if (outKey) {
      outNTuple = outKey->ReadObject<RNTuple>();
      if (!outNTuple) {
         Error("RNTuple::Merge", "Output file already has key, but not of type RNTuple!");
         return -1;
      }
      // In principle, we should already be working on the RNTuple object from the output file, but just continue with
      // pointer we just got.
   }

   // The "fast" option is present if and only if we don't want to change compression.
   const int compression =
      mergeInfo->fOptions.Contains("fast") ? kUnknownCompressionSettings : outFile->GetCompressionSettings();

   RNTupleWriteOptions writeOpts;
   writeOpts.SetUseBufferedWrite(false);
   if (compression != kUnknownCompressionSettings)
      writeOpts.SetCompression(compression);
   auto destination = std::make_unique<RPageSinkFile>(ntupleName, *outFile, writeOpts);

   // If we already have an existing RNTuple, copy over its descriptor to support incremental merging
   if (outNTuple) {
      auto source = RPageSourceFile::CreateFromAnchor(*outNTuple);
      source->Attach();
      auto desc = source->GetSharedDescriptorGuard();
      destination->InitFromDescriptor(desc.GetRef());
   }

   // The remaining entries are the input files
   std::vector<std::unique_ptr<RPageSourceFile>> sources;
   std::vector<RPageSource *> sourcePtrs;

   while (const auto &pitr = itr()) {
      TFile *inFile = dynamic_cast<TFile *>(pitr);
      RNTuple *anchor = inFile ? inFile->Get<RNTuple>(ntupleName.c_str()) : nullptr;
      if (!anchor) {
         Error("RNTuple::Merge", "Failed to retrieve RNTuple anchor named '%s' from file '%s'", ntupleName.c_str(),
               inFile->GetName());
         return -1;
      }
      sources.push_back(RPageSourceFile::CreateFromAnchor(*anchor));
   }

   // Interface conversion
   sourcePtrs.reserve(sources.size());
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   // Now merge
   RNTupleMerger merger;
   RNTupleMergeOptions options;
   options.fCompressionSettings = compression;
   merger.Merge(sourcePtrs, *destination, options).ThrowOnError();

   // Provide the caller with a merged anchor object (even though we've already
   // written it).
   *this = *outFile->Get<RNTuple>(ntupleName.c_str());

   return 0;
} catch (const RException &ex) {
   Error("RNTuple::Merge", "Exception thrown while merging: %s", ex.what());
   return -1;
}

// Functor used to change the compression of a page to `options.fCompressionSettings`.
struct RNTupleMerger::RChangeCompressionFunc {
   size_t pageIdx;
   size_t pageBufferBaseIdx;
   size_t checksumSize;
   int colRangeCompressionSettings;
   DescriptorId_t outputColumnId;

   const RColumnElementBase &srcColElement;
   const RColumnElementBase &dstColElement;
   const RClusterDescriptor::RPageRange::RPageInfo &pageInfo;
   const RNTupleMergeOptions &options;

   RPageStorage::RSealedPage &sealedPage;
   std::vector<std::unique_ptr<unsigned char[]>> &sealedPageBuffers;
   RPageSource &source;

   void operator()() const
   {
      const auto uncompressedSize = srcColElement.GetSize() * sealedPage.GetNElements();
      auto &buffer = sealedPageBuffers[pageBufferBaseIdx + pageIdx];
      buffer = std::make_unique<unsigned char[]>(uncompressedSize + checksumSize);

      auto page = source.UnsealPage(sealedPage, srcColElement, outputColumnId).Unwrap();
      RPageSink::RSealPageConfig sealConf;
      sealConf.fElement = &dstColElement;
      sealConf.fPage = &page;
      sealConf.fBuffer = buffer.get();
      sealConf.fCompressionSetting = options.fCompressionSettings;
      auto resealedPage = RPageSink::SealPage(sealConf);
      sealedPage = resealedPage;
   }
};

namespace {
struct RDescriptorsComparison {
   std::vector<const RFieldDescriptor *> fExtraDstFields;
   std::vector<const RFieldDescriptor *> fExtraSrcFields;
   std::vector<const RFieldDescriptor *> fCommonFields;
};

struct RColumnOutInfo {
   DescriptorId_t fColumnId;
   EColumnType fColumnType;
};

// { fully.qualified.fieldName.colInputId => colOutputInfo }
using ColumnIdMap_t = std::unordered_map<std::string, RColumnOutInfo>;

struct RColumnInfo {
   std::string fColumnName;
   DescriptorId_t fInputId;
   DescriptorId_t fOutputId;
   EColumnType fColumnType;
   const RFieldDescriptor *fParentField;
};

struct RColumnInfoGroup {
   std::vector<RColumnInfo> fExtraDstColumns;
   std::vector<RColumnInfo> fCommonColumns;
};
} // namespace

/// Compares the top level fields of `dst` and `src` and determines whether they can be merged or not.
/// In addition, returns the differences between `dst` and `src`'s structures
static RResult<RDescriptorsComparison>
CompareDescriptorStructure(const RNTupleDescriptor &dst, const RNTupleDescriptor &src)
{
   // Cases:
   // 1. dst == src
   // 2. dst has fields that src hasn't
   // 3. src has fields that dst hasn't
   // 4. dst and src have fields that differ (compatible or incompatible)

   std::vector<std::string> errors;
   RDescriptorsComparison res;

   struct RCommonField {
      const RFieldDescriptor *fDst;
      const RFieldDescriptor *fSrc;
   };
   std::vector<RCommonField> commonFields;

   for (const auto &dstField : dst.GetTopLevelFields()) {
      if (const auto *srcField = src.FindFieldDescriptor(dstField.GetFieldName()))
         commonFields.push_back({&dstField, srcField});
      else
         res.fExtraDstFields.emplace_back(&dstField);
   }
   for (const auto &srcField : src.GetTopLevelFields()) {
      if (!dst.FindFieldDescriptor(srcField.GetFieldName()))
         res.fExtraSrcFields.push_back(&srcField);
   }

   // Check compatibility of common fields
   for (const auto &field : commonFields) {
      const auto &srcFdName = field.fSrc->GetFieldName();

      // Require that fields are both projected or both not projected
      bool projCompatible = field.fSrc->IsProjectedField() == field.fDst->IsProjectedField();
      if (!projCompatible) {
         std::stringstream ss;
         ss << "Field `" << srcFdName << "` is incompatible with previously-seen field with that name because the "
            << (field.fSrc->IsProjectedField() ? "new" : "old") << " one is projected and the other isn't";
         errors.push_back(ss.str());
      } else if (field.fSrc->IsProjectedField()) {
         // if both fields are projected, verify that they point to the same real field
         const auto srcId = field.fSrc->GetProjectionSourceId();
         const auto dstId = field.fDst->GetProjectionSourceId();
         if (srcId != dstId) {
            std::stringstream ss;
            ss << "Field `" << srcFdName
               << "` is projected to a different field than a previously-seen field with the same name (old: " << dstId
               << ", new: " << srcId << ")";
            errors.push_back(ss.str());
         }
      }

      // Require that fields types match
      // TODO(gparolini): allow non-identical but compatible types
      const auto &srcTyName = field.fSrc->GetTypeName();
      const auto &dstTyName = field.fDst->GetTypeName();
      if (srcTyName != dstTyName) {
         std::stringstream ss;
         ss << "Field `" << srcFdName
            << "` has a type incompatible with a previously-seen field with the same name: (old: " << dstTyName
            << ", new: " << srcTyName << ")";
         errors.push_back(ss.str());
      }

      const auto srcTyChk = field.fSrc->GetTypeChecksum();
      const auto dstTyChk = field.fDst->GetTypeChecksum();
      if (srcTyChk && dstTyChk && *srcTyChk != *dstTyChk) {
         std::stringstream ss;
         ss << "Field `" << field.fSrc->GetFieldName()
            << "` has a different type checksum than previously-seen field with the same name";
         errors.push_back(ss.str());
      }

      const auto srcTyVer = field.fSrc->GetTypeVersion();
      const auto dstTyVer = field.fDst->GetTypeVersion();
      if (srcTyVer != dstTyVer) {
         std::stringstream ss;
         ss << "Field `" << field.fSrc->GetFieldName()
            << "` has a different type version than previously-seen field with the same name (old: " << dstTyVer
            << ", new: " << srcTyVer << ")";
         errors.push_back(ss.str());
      }
   }

   std::string errMsg;
   for (const auto &err : errors)
      errMsg += std::string("\n  * ") + err;

   if (errMsg.length())
      return R__FAIL(errMsg);

   res.fCommonFields.reserve(commonFields.size());
   for (const auto &[_, srcField] : commonFields) {
      res.fCommonFields.emplace_back(srcField);
   }

   return RResult(res);
}

// Applies late model extension to `destination`, adding all `newFields` to it.
static void ExtendDestinationModel(std::span<const RFieldDescriptor *> newFields, RPageSink &destination,
                                   RNTupleModel &dstModel, NTupleSize_t nDstEntries)
{
   assert(newFields.size() > 0); // no point in calling this with 0 new cols

   dstModel.Unfreeze();
   RNTupleModelChangeset changeset{dstModel};

   std::string msg = "destination doesn't contain field";
   if (newFields.size() > 1)
      msg += 's';
   msg += ' ';
   msg += std::accumulate(newFields.begin(), newFields.end(), std::string{}, [](const auto &acc, const auto *field) {
      return acc + (acc.length() ? ", " : "") + '`' + field->GetFieldName() + '`';
   });
   Info("RNTuple::Merge", "%s: adding %s to the destination model (entry #%lu).", msg.c_str(),
        (newFields.size() > 1 ? "them" : "it"), nDstEntries);

   changeset.fAddedFields.reserve(newFields.size());
   for (const auto *fieldDesc : newFields) {
      auto field = fieldDesc->CreateField(destination.GetDescriptor());
      if (fieldDesc->IsProjectedField())
         changeset.fAddedProjectedFields.emplace_back(field.get());
      else
         changeset.fAddedFields.emplace_back(field.get());
      changeset.fModel.AddField(std::move(field));
   }
   dstModel.Freeze();
   destination.UpdateSchema(changeset, nDstEntries);
}

// Merges all columns appearing both in the source and destination RNTuples, just copying them if their
// compression matches ("fast merge") or by unsealing and resealing them with the proper compression.
static void MergeCommonColumns(RClusterPool &clusterPool, DescriptorId_t clusterId,
                               std::span<RColumnInfo> commonColumns, RCluster::ColumnSet_t commonColumnSet,
                               std::optional<TTaskGroup> &taskGroup,
                               std::deque<RPageStorage::SealedPageSequence_t> &sealedPagesV,
                               std::vector<RPageStorage::RSealedPageGroup> &sealedPageGroups,
                               std::vector<std::unique_ptr<unsigned char[]>> &sealedPageBuffers, RPageSource &source,
                               const RNTupleDescriptor &srcDescriptor, const RNTupleMergeOptions &options)
{
   assert(commonColumns.size() == commonColumnSet.size());
   if (commonColumns.empty())
      return;

   const RCluster *cluster = clusterPool.GetCluster(clusterId, commonColumnSet);
   if (!cluster)
      return;

   const auto &clusterDesc = srcDescriptor.GetClusterDescriptor(clusterId);

   for (const auto &column : commonColumns) {
      const auto &columnId = column.fInputId;
      R__ASSERT(clusterDesc.ContainsColumn(columnId));

      const auto &columnDesc = srcDescriptor.GetColumnDescriptor(columnId);
      const auto srcColElement = RColumnElementBase::Generate(columnDesc.GetType());
      const auto dstColElement = RColumnElementBase::Generate(column.fColumnType);

      // Now get the pages for this column in this cluster
      const auto &pages = clusterDesc.GetPageRange(columnId);

      RPageStorage::SealedPageSequence_t sealedPages;
      sealedPages.resize(pages.fPageInfos.size());

      // Each column range potentially has a distinct compression settings
      const auto colRangeCompressionSettings = clusterDesc.GetColumnRange(columnId).fCompressionSettings;
      const bool needsCompressionChange = options.fCompressionSettings != kUnknownCompressionSettings &&
                                          colRangeCompressionSettings != options.fCompressionSettings;

      if (needsCompressionChange)
         Info("RNTuple::Merge", "Column %s: changing source compression from %d to %d", column.fColumnName.c_str(),
              colRangeCompressionSettings, options.fCompressionSettings);

      // If the column range is already uncompressed we don't need to allocate any new buffer, so we don't
      // bother reserving memory for them.
      size_t pageBufferBaseIdx = sealedPageBuffers.size();
      if (colRangeCompressionSettings != 0)
         sealedPageBuffers.resize(sealedPageBuffers.size() + pages.fPageInfos.size());

      // Loop over the pages
      std::uint64_t pageIdx = 0;
      for (const auto &pageInfo : pages.fPageInfos) {
         assert(pageIdx < sealedPages.size());
         assert(sealedPageBuffers.size() == 0 || pageIdx < sealedPageBuffers.size());

         ROnDiskPage::Key key{columnId, pageIdx};
         auto onDiskPage = cluster->GetOnDiskPage(key);

         const auto checksumSize = pageInfo.fHasChecksum * RPageStorage::kNBytesPageChecksum;
         RPageStorage::RSealedPage &sealedPage = sealedPages[pageIdx];
         sealedPage.SetNElements(pageInfo.fNElements);
         sealedPage.SetHasChecksum(pageInfo.fHasChecksum);
         sealedPage.SetBufferSize(pageInfo.fLocator.fBytesOnStorage + checksumSize);
         sealedPage.SetBuffer(onDiskPage->GetAddress());
         // TODO(gparolini): more graceful error handling (skip the page?)
         sealedPage.VerifyChecksumIfEnabled().ThrowOnError();
         R__ASSERT(onDiskPage && (onDiskPage->GetSize() == sealedPage.GetBufferSize()));

         if (needsCompressionChange) {
            RNTupleMerger::RChangeCompressionFunc compressTask{
               pageIdx,          pageBufferBaseIdx, checksumSize,      colRangeCompressionSettings,
               column.fOutputId, *srcColElement,    *dstColElement,    pageInfo,
               options,          sealedPage,        sealedPageBuffers, source};

            if (taskGroup)
               taskGroup->Run(compressTask);
            else
               compressTask();
         }

         ++pageIdx;

      } // end of loop over pages

      if (taskGroup)
         taskGroup->Wait();

      sealedPagesV.push_back(std::move(sealedPages));
      sealedPageGroups.emplace_back(column.fOutputId, sealedPagesV.back().cbegin(), sealedPagesV.back().cend());
   } // end loop over common columns
}

// Generates default values for columns that are not present in the current source RNTuple
// but are present in the destination's schema.
static void GenerateExtraDstColumns(size_t nClusterEntries, std::span<RColumnInfo> extraDstColumns,
                                    std::deque<RPageStorage::SealedPageSequence_t> &sealedPagesV,
                                    std::vector<RPageStorage::RSealedPageGroup> &sealedPageGroups,
                                    std::vector<std::unique_ptr<unsigned char[]>> &sealedPageBuffers,
                                    const RNTupleDescriptor &srcDescriptor, const RNTupleDescriptor &dstDescriptor)
{
   for (const auto &column : extraDstColumns) {
      const auto &columnId = column.fInputId;
      const auto &columnDesc = dstDescriptor.GetColumnDescriptor(columnId);

      // Check if this column is a child of a Collection or a Variant. If so, it has no data
      // and can be skipped.
      const RFieldDescriptor *field = column.fParentField;
      bool skipColumn = false;
      for (auto parentId = field->GetParentId(); parentId != kInvalidDescriptorId;) {
         const RFieldDescriptor &parent = srcDescriptor.GetFieldDescriptor(parentId);
         if (parent.GetStructure() == ENTupleStructure::kCollection ||
             parent.GetStructure() == ENTupleStructure::kVariant) {
            skipColumn = true;
            break;
         }
         parentId = parent.GetParentId();
      }
      if (skipColumn)
         continue;

      const auto structure = field->GetStructure();

      if (structure == ENTupleStructure::kUnsplit) {
         Warning("RNTuple::Merge",
                 "Found a column associated to unsplit field %s. Merging unsplit fields is not supported, therefore "
                 "this column will be skipped.",
                 field->GetFieldName().c_str());
         continue;
      }

      // NOTE: we cannot have a Record here because it has no associated columns.
      R__ASSERT(structure == ENTupleStructure::kCollection || structure == ENTupleStructure::kVariant ||
                structure == ENTupleStructure::kLeaf);

      const auto colElement = RColumnElementBase::Generate(columnDesc.GetType());
      const auto nRepetitions =
         (structure == ENTupleStructure::kCollection || field->GetNRepetitions() > 0) ? field->GetNRepetitions() : 1;
      const auto nElements = nClusterEntries * nRepetitions;
      const auto bytesOnStorage = colElement->GetPackedSize(nElements);
      constexpr auto kPageSizeLimit = 256 * 1024;
      // TODO(gparolini): consider coalescing the last page if its size is less than some threshold
      const size_t nPages = bytesOnStorage / kPageSizeLimit + !!(bytesOnStorage % kPageSizeLimit);
      for (size_t i = 0; i < nPages; ++i) {
         const auto pageSize = (i < nPages - 1) ? kPageSizeLimit : bytesOnStorage - kPageSizeLimit * (nPages - 1);
         auto &buffer = sealedPageBuffers.emplace_back(new unsigned char[pageSize]);

         RPageStorage::RSealedPage sealedPage;
         sealedPage.SetHasChecksum(true);
         sealedPage.SetNElements(nElements);
         sealedPage.SetBufferSize(pageSize);
         sealedPage.SetBuffer(buffer.get());

         memset(buffer.get(), 0, pageSize);

         sealedPagesV.push_back({sealedPage});
      }

      sealedPageGroups.emplace_back(column.fOutputId, sealedPagesV.back().cbegin(), sealedPagesV.back().cend());
   }
}

// Iterates over all clusters of `source` and merges their pages into `destination`.
// It is assumed that all columns in `commonColumns` are present (and compatible) in both the source and
// the destination's schemas.
// The pages may be "fast-merged" (i.e. simply copied with no decompression/recompression) if the target
// compression is unspecified or matches the original compression settings.
static void MergeSourceClusters(RPageSink &destination, RPageSource &source, NTupleSize_t &nDstEntries,
                                const RNTupleDescriptor &srcDescriptor, std::span<RColumnInfo> commonColumns,
                                std::span<RColumnInfo> extraDstColumns, std::optional<TTaskGroup> &taskGroup,
                                const RNTupleMergeOptions &options)
{
   RClusterPool clusterPool{source};

   // Convert columns to a ColumnSet for the ClusterPool query
   RCluster::ColumnSet_t commonColumnSet;
   commonColumnSet.reserve(commonColumns.size());
   for (const auto &column : commonColumns)
      commonColumnSet.emplace(column.fInputId);

   RCluster::ColumnSet_t extraDstColumnSet;
   extraDstColumnSet.reserve(extraDstColumns.size());
   for (const auto &column : extraDstColumns)
      extraDstColumnSet.emplace(column.fInputId);

   // Loop over all clusters in this file.
   // descriptor->GetClusterIterable() doesn't guarantee any specific order, so we explicitly
   // request the first cluster.
   DescriptorId_t clusterId = srcDescriptor.FindClusterId(0, 0);
   while (clusterId != kInvalidDescriptorId) {
      const auto &clusterDesc = srcDescriptor.GetClusterDescriptor(clusterId);
      const auto nClusterEntries = clusterDesc.GetNEntries();
      if (nClusterEntries == 0)
         continue;

      // We use a std::deque so that references to the contained SealedPageSequence_t, and its iterators, are
      // never invalidated.
      std::deque<RPageStorage::SealedPageSequence_t> sealedPagesV;
      std::vector<RPageStorage::RSealedPageGroup> sealedPageGroups;
      std::vector<std::unique_ptr<unsigned char[]>> sealedPageBuffers;

      if (!commonColumnSet.empty()) {
         MergeCommonColumns(clusterPool, clusterId, commonColumns, commonColumnSet, taskGroup, sealedPagesV,
                            sealedPageGroups, sealedPageBuffers, source, srcDescriptor, options);
      }

      if (!extraDstColumnSet.empty()) {
         GenerateExtraDstColumns(nClusterEntries, extraDstColumns, sealedPagesV, sealedPageGroups, sealedPageBuffers,
                                 srcDescriptor, destination.GetDescriptor());
      }

      // Commit the pages and the clusters
      destination.CommitSealedPageV(sealedPageGroups);
      destination.CommitCluster(nClusterEntries);
      nDstEntries += nClusterEntries;

      // Go to the next cluster
      clusterId = srcDescriptor.FindNextClusterId(clusterId);
   }

   // TODO(gparolini): when we get serious about huge file support (>~ 100GB) we might want to check here
   // the size of the running page list and commit a cluster group when it exceeds some threshold,
   // which would prevent the page list from getting too large.
   // However, as of today, we aren't really handling such huge files, and even relatively big ones
   // such as the CMS dataset have a page list size of about only 2 MB.
   // So currently we simply merge all cluster groups into one.
}

// Given a field, fill `columns` and `colIdMap` with information about all columns belonging to it and its subfields.
// `colIdMap` is used to map matching columns from different sources to the same output column in the destination.
// We match columns by their "fully qualified name", which is the concatenation of their ancestor fields' names
// and the column index.
// By this point, since we called `CompareDescriptorStructures()` earlier, we should be guaranteed that two matching
// columns will have at least compatible representations.
static void AddColumnsFromField(std::vector<RColumnInfo> &columns, ColumnIdMap_t &colIdMap,
                                const RNTupleDescriptor &srcDesc, const RNTupleDescriptor &dstDesc,
                                const RFieldDescriptor &fieldDesc, const std::string &prefix = "")
{
   std::string name = prefix + '.' + fieldDesc.GetFieldName();

   const auto &columnIds = fieldDesc.GetLogicalColumnIds();
   columns.reserve(columns.size() + columnIds.size());
   for (const auto &columnId : columnIds) {
      const auto &srcColumn = srcDesc.GetColumnDescriptor(columnId);
      RColumnInfo info;
      info.fColumnName = name + '.' + std::to_string(srcColumn.GetIndex());
      info.fInputId = columnId;
      info.fParentField = &fieldDesc;

      if (auto it = colIdMap.find(info.fColumnName); it != colIdMap.end()) {
         info.fOutputId = it->second.fColumnId;
         info.fColumnType = it->second.fColumnType;
      } else {
         info.fOutputId = colIdMap.size();
         // NOTE(gparolini): map the type of src column to the type of dst column.
         // This mapping is only relevant for common columns and it's done to ensure we keep a consistent
         // on-disk representation of the same column.
         // This is also important to do for first source when it is used to generate the destination sink,
         // because even in that case their column representations may differ.
         // e.g. if the destination has a different compression than the source, an integer column might be
         // zigzag-encoded in the source but not in the destination.
         const auto &dstColumn = (&dstDesc == &srcDesc) ? srcColumn : dstDesc.GetColumnDescriptor(columnId);
         info.fColumnType = dstColumn.GetType();
         colIdMap[info.fColumnName] = {info.fOutputId, info.fColumnType};
      }
      columns.emplace_back(info);
   }

   for (const auto &field : srcDesc.GetFieldIterable(fieldDesc))
      AddColumnsFromField(columns, colIdMap, srcDesc, dstDesc, field, name);
}

// Converts the fields comparison data to the corresponding column information.
// While doing so, it collects such information in `colIdMap`, which is used by later calls to this function
// to map already-seen column names to their chosen outputId, type and so on.
static RColumnInfoGroup GatherColumnInfos(const RDescriptorsComparison &descCmp, const RNTupleDescriptor &dstDesc,
                                          const RNTupleDescriptor &srcDesc, ColumnIdMap_t &colIdMap)
{
   RColumnInfoGroup res;
   for (const RFieldDescriptor *field : descCmp.fExtraDstFields) {
      AddColumnsFromField(res.fExtraDstColumns, colIdMap, dstDesc, dstDesc, *field);
   }
   for (const auto *field : descCmp.fCommonFields) {
      AddColumnsFromField(res.fCommonColumns, colIdMap, srcDesc, dstDesc, *field);
   }
   return res;
}

RResult<void>
RNTupleMerger::Merge(std::span<RPageSource *> sources, RPageSink &destination, const RNTupleMergeOptions &options) const
{
   auto &dstDescriptor = destination.GetDescriptor();

   std::unique_ptr<RNTupleModel> model; // used to initialize the schema of the output RNTuple
   NTupleSize_t nDstEntries = 0;
   std::vector<RColumnInfo> columns;
   ColumnIdMap_t columnIdMap;
   std::optional<TTaskGroup> taskGroup;
#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled())
      taskGroup = TTaskGroup();
#endif

#define SKIP_OR_ABORT(errMsg)                                                        \
   do {                                                                              \
      if (options.fErrBehavior == ENTupleMergeErrBehavior::kSkip) {                  \
         Warning("RNTuple::Merge", "Skipping RNTuple due to: %s", (errMsg).c_str()); \
         continue;                                                                   \
      } else {                                                                       \
         return R__FAIL(errMsg);                                                     \
      }                                                                              \
   } while (0)

   // Merge main loop
   for (RPageSource *source : sources) {
      source->Attach();
      auto srcDescriptor = source->GetSharedDescriptorGuard();

      // Create sink from the input model if not initialized
      if (!destination.IsInitialized()) {
         model = srcDescriptor->CreateModel();
         destination.Init(*model);
      }

      for (const auto &extraTypeInfoDesc : srcDescriptor->GetExtraTypeInfoIterable())
         destination.UpdateExtraTypeInfo(extraTypeInfoDesc);

      auto descCmpRes = CompareDescriptorStructure(dstDescriptor, srcDescriptor.GetRef());
      if (!descCmpRes) {
         SKIP_OR_ABORT(
            std::string("Source RNTuple will be skipped due to incompatible schema with the destination:\n") +
            descCmpRes.GetError()->GetReport());
      }
      auto descCmp = descCmpRes.Unwrap();

      // If the current source is missing some fields and we're not in Union mode, error
      if (options.fMergingMode != ENTupleMergingMode::kUnion && !descCmp.fExtraDstFields.empty()) {
         std::string msg = "Source RNTuple is missing the following fields:";
         for (const auto *field : descCmp.fExtraDstFields) {
            msg += "\n  " + field->GetFieldName() + " : " + field->GetTypeName();
         }
         SKIP_OR_ABORT(msg);
      }

      // handle extra src fields
      if (descCmp.fExtraSrcFields.size()) {
         if (options.fMergingMode == ENTupleMergingMode::kUnion) {
            // late model extension for all fExtraSrcFields in Union mode
            ExtendDestinationModel(descCmp.fExtraSrcFields, destination, *model, nDstEntries);
            descCmp.fCommonFields.insert(descCmp.fCommonFields.end(), descCmp.fExtraSrcFields.begin(),
                                         descCmp.fExtraSrcFields.end());
         } else if (options.fMergingMode == ENTupleMergingMode::kStrict) {
            // If the current source has extra fields and we're in Strict mode, error
            std::string msg = "Source RNTuple has extra fields that the destination RNTuple doesn't have:";
            for (const auto *field : descCmp.fExtraSrcFields) {
               msg += "\n  " + field->GetFieldName() + " : " + field->GetTypeName();
            }
            SKIP_OR_ABORT(msg);
         }
      }

      // handle extra dst fields & common fields
      auto columnInfos = GatherColumnInfos(descCmp, dstDescriptor, srcDescriptor.GetRef(), columnIdMap);
      MergeSourceClusters(destination, *source, nDstEntries, srcDescriptor.GetRef(), columnInfos.fCommonColumns,
                          columnInfos.fExtraDstColumns, taskGroup, options);
   } // end loop over sources

   // Commit the output
   destination.CommitClusterGroup();
   destination.CommitDataset();

   return RResult<void>::Success();
}
