/// \file RNTupleMerger.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>, Max Orok <maxwellorok@gmail.com>, Alaettin Serhan Mete <amete@anl.gov>
/// \date 2020-07-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

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
#include <TROOT.h>
#include <TFileMergeInfo.h>
#include <TError.h>
#include <TFile.h>
#include <TKey.h>

#include <deque>

Long64_t ROOT::Experimental::RNTuple::Merge(TCollection *inputs, TFileMergeInfo *mergeInfo)
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

   // The "fast" options is present if and only if we don't want to change compression.
   const int compression =
      mergeInfo->fOptions.Contains("fast") ? kUnknownCompressionSettings : outFile->GetCompressionSettings();

   RNTupleWriteOptions writeOpts;
   writeOpts.SetUseBufferedWrite(false);
   if (compression != kUnknownCompressionSettings)
      writeOpts.SetCompression(compression);
   auto destination = std::make_unique<Internal::RPageSinkFile>(ntupleName, *outFile, writeOpts);

   // If we already have an existing RNTuple, copy over its descriptor to support incremental merging
   if (outNTuple) {
      auto source = Internal::RPageSourceFile::CreateFromAnchor(*outNTuple);
      source->Attach();
      auto desc = source->GetSharedDescriptorGuard();
      destination->InitFromDescriptor(desc.GetRef());
   }

   // The remaining entries are the input files
   std::vector<std::unique_ptr<Internal::RPageSourceFile>> sources;
   std::vector<Internal::RPageSource *> sourcePtrs;

   while (const auto &pitr = itr()) {
      TFile *inFile = dynamic_cast<TFile *>(pitr);
      RNTuple *anchor = inFile ? inFile->Get<RNTuple>(ntupleName.c_str()) : nullptr;
      if (!anchor) {
         Error("RNTuple::Merge", "Failed to retrieve RNTuple anchor named '%s' from file '%s'", ntupleName.c_str(),
               inFile->GetName());
         return -1;
      }
      sources.push_back(Internal::RPageSourceFile::CreateFromAnchor(*anchor));
   }

   // Interface conversion
   sourcePtrs.reserve(sources.size());
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   // Now merge
   Internal::RNTupleMerger merger;
   Internal::RNTupleMergeOptions options;
   options.fCompressionSettings = compression;
   merger.Merge(sourcePtrs, *destination, options);

   // Provide the caller with a merged anchor object (even though we've already
   // written it).
   *this = *outFile->Get<RNTuple>(ntupleName.c_str());

   return 0;
} catch (const RException &ex) {
   Error("RNTuple::Merge", "Exception thrown while merging: %s", ex.what());
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
void ROOT::Experimental::Internal::RNTupleMerger::ValidateColumns(std::vector<RColumnInfo> &columns) const
{
   // First ensure that we have the same number of columns
   if (fOutputIdMap.size() != columns.size()) {
      throw RException(R__FAIL("Columns between sources do NOT match"));
   }
   // Then ensure that we have the same names of columns and assign the ids
   for (auto &column : columns) {
      try {
         column.fColumnOutputId = fOutputIdMap.at(column.fColumnNameTypeAndVersion);
      } catch (const std::out_of_range &) {
         throw RException(R__FAIL("Column NOT found in the first source w/ name " + column.fColumnNameTypeAndVersion));
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
size_t ROOT::Experimental::Internal::RNTupleMerger::CollectColumns(const RNTupleDescriptor &descriptor,
                                                                   RNTupleMergingMode mergingMode,
                                                                   std::vector<RColumnInfo> &columns)
{
   // Here we recursively find the columns and fill the RColumnInfo vector
   columns.clear();
   const auto nNewCols = AddColumnsFromField(columns, descriptor, descriptor.GetFieldZero());

   // Then we either build the internal map (first source) or validate the columns against it (remaning sources)
   // In either case, we also assign the output ids here

   if (mergingMode == RNTupleMergingMode::kUnion) {
      // Assign new ids to new columns
      for (size_t i = 0; i < nNewCols; ++i) {
         auto &column = columns[i];
         column.fColumnOutputId = fOutputIdMap.size();
         assert(fOutputIdMap.find(column.fColumnNameTypeAndVersion) == fOutputIdMap.end());
         fOutputIdMap[column.fColumnNameTypeAndVersion] = column.fColumnOutputId;
      }
      // Assign existing ids to old columns
      for (size_t i = nNewCols; i < columns.size(); ++i) {
         auto &column = columns[i];
         auto id = fOutputIdMap.find(column.fColumnNameTypeAndVersion);
         assert(id != fOutputIdMap.end());
         column.fColumnOutputId = id->second;
      }
   } else {
      if (fOutputIdMap.empty()) {
         for (auto &column : columns) {
            column.fColumnOutputId = fOutputIdMap.size();
            fOutputIdMap[column.fColumnNameTypeAndVersion] = column.fColumnOutputId;
         }
      } else {
         ValidateColumns(columns);
      }
   }

   return nNewCols;
}

////////////////////////////////////////////////////////////////////////////////
size_t ROOT::Experimental::Internal::RNTupleMerger::AddColumnsFromField(std::vector<RColumnInfo> &columns,
                                                                        const RNTupleDescriptor &desc,
                                                                        const RFieldDescriptor &fieldDesc,
                                                                        const std::string &prefix) const
{
   size_t nNewCols = 0;
   for (const auto &field : desc.GetFieldIterable(fieldDesc)) {
      std::string name = prefix + field.GetFieldName() + ".";
      const std::string typeAndVersion = field.GetTypeName() + "." + std::to_string(field.GetTypeVersion());
      auto columnIter = desc.GetColumnIterable(field);
      columns.reserve(columns.size() + columnIter.count());
      for (const auto &column : columnIter) {
         RColumnInfo info{name + std::to_string(column.GetIndex()), typeAndVersion, column.GetPhysicalId(),
                          kInvalidDescriptorId};

         // Output columns like this:
         //    [new_columns|old_columns]
         // where both old and new columns preserve the original order.
         if (fOutputIdMap.find(info.fColumnNameTypeAndVersion) != fOutputIdMap.end()) {
            columns.emplace_back(info);
         } else {
            columns.insert(columns.begin() + nNewCols, info);
            ++nNewCols;
         }
      }
      nNewCols += AddColumnsFromField(columns, desc, field, name);
   }

   return nNewCols;
}

void ROOT::Experimental::Internal::RNTupleMerger::ExtendOutputModel(RNTupleModel &model, std::span<RColumnInfo> newCols,
                                                                    int nDstEntries,
                                                                    const RNTupleDescriptor &descriptor,
                                                                    RPageSink &destination) const
{
   assert(newCols.size() > 0); // no point in calling this with 0 new cols

   model.Unfreeze();
   Internal::RNTupleModelChangeset changeset{model};

   std::string msg = "[ INFO ] destination doesn't contain column";
   if (newCols.size() > 1)
      msg += 's';
   msg += ' ';
   msg = std::accumulate(newCols.begin(), newCols.end(), msg, [](const std::string &acc, const RColumnInfo &col) {
      return acc + ", " + col.fColumnNameTypeAndVersion;
   });
   std::cerr << msg << ": adding it to the destination model.\n";

   changeset.fAddedFields.reserve(newCols.size());
   for (const auto &column : newCols) {
      const auto columnId = column.fColumnInputId;
      const auto &columnDesc = descriptor.GetColumnDescriptor(columnId);
      const auto &fieldId = columnDesc.GetFieldId();
      const auto &fieldDesc = descriptor.GetFieldDescriptor(fieldId);
      auto field = fieldDesc.CreateField(descriptor);
      // TODO(gparolini): handle projected fields
      changeset.fAddedFields.emplace_back(field.get());
      changeset.fModel.AddField(std::move(field));
   }
   model.Freeze();
   destination.UpdateSchema(changeset, nDstEntries);
}

////////////////////////////////////////////////////////////////////////////////
void ROOT::Experimental::Internal::RNTupleMerger::Merge(std::span<RPageSource *> sources, RPageSink &destination,
                                                        const RNTupleMergeOptions &options)
{
   std::vector<RColumnInfo> srcColumns;
   RCluster::ColumnSet_t columnSet;

   if (destination.IsInitialized()) {
      auto nNewCols = CollectColumns(destination.GetDescriptor(), options.fMergingMode, srcColumns);
      assert(nNewCols == srcColumns.size());
   }

   std::unique_ptr<RNTupleModel> model; // used to initialize the schema of the output RNTuple
   std::optional<TTaskGroup> taskGroup;
#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled())
      taskGroup = TTaskGroup();
#endif

   auto nDstEntries = 0;

   // Append the sources to the destination one-by-one
   for (const auto &source : sources) {
      source->Attach();

      RClusterPool clusterPool{*source};

      // Get a handle on the descriptor (metadata)
      auto descriptor = source->GetSharedDescriptorGuard();

      // Collect all the srcColumns
      size_t nNewCols = CollectColumns(descriptor.GetRef(), options.fMergingMode, srcColumns);

      // If this source contains columns that weren't known so far, perform late model extension
      // of the destination model.
      if (options.fMergingMode == RNTupleMergingMode::kUnion && model && nNewCols > 0) {
         std::span<RColumnInfo> newCols{srcColumns.data(), nNewCols};
         ExtendOutputModel(*model, srcColumns, nDstEntries, descriptor.GetRef(), destination);
      }

      columnSet.clear();
      columnSet.reserve(srcColumns.size());
      for (const auto &column : srcColumns)
         columnSet.emplace(column.fColumnInputId);

      // Create sink from the input model if not initialized
      if (!destination.IsInitialized()) {
         model = descriptor->CreateModel();
         destination.Init(*model.get());
      }

      for (const auto &extraTypeInfoDesc : descriptor->GetExtraTypeInfoIterable()) {
         destination.UpdateExtraTypeInfo(extraTypeInfoDesc);
      }

      // Make sure the source contains events to be merged
      if (source->GetNEntries() == 0) {
         continue;
      }

      // Now loop over all clusters in this file
      // descriptor->GetClusterIterable() doesn't guarantee any specific order...
      // Find the first cluster id and iterate from there...
      auto clusterId = descriptor->FindClusterId(0, 0);

      while (clusterId != ROOT::Experimental::kInvalidDescriptorId) {
         auto *cluster = clusterPool.GetCluster(clusterId, columnSet);
         assert(cluster);
         const auto &clusterDesc = descriptor->GetClusterDescriptor(clusterId);

         // We use a std::deque so that references to the contained SealedPageSequence_t, and its iterators, are never
         // invalidated.
         std::deque<RPageStorage::SealedPageSequence_t> sealedPagesV;
         std::vector<RPageStorage::RSealedPageGroup> sealedPageGroups;
         std::vector<std::unique_ptr<unsigned char[]>> sealedPageBuffers;

         for (const auto &column : srcColumns) {

            assert(column.fColumnOutputId != (DescriptorId_t)-1);

            const auto columnId = column.fColumnInputId;
            const auto &columnDesc = descriptor->GetColumnDescriptor(columnId);

            // See if this cluster contains this column
            // if not, there is nothing to read/do..
            if (!clusterDesc.ContainsColumn(columnId)) {
               continue;
            }

            const auto colElement = RColumnElementBase::Generate(columnDesc.GetType());

            // Now get the pages for this column in this cluster
            const auto &pages = clusterDesc.GetPageRange(columnId);

            RPageStorage::SealedPageSequence_t sealedPages;
            sealedPages.resize(pages.fPageInfos.size());

            // Each column range potentially has a distinct compression settings
            const auto colRangeCompressionSettings = clusterDesc.GetColumnRange(columnId).fCompressionSettings;
            const bool needsCompressionChange = options.fCompressionSettings != kUnknownCompressionSettings &&
                                                colRangeCompressionSettings != options.fCompressionSettings;

            // If the column range is already uncompressed we don't need to allocate any new buffer, so we don't
            // bother reserving memory for them.
            size_t pageBufferBaseIdx = sealedPageBuffers.size();
            if (colRangeCompressionSettings != 0)
               sealedPageBuffers.resize(sealedPageBuffers.size() + pages.fPageInfos.size());

            std::uint64_t pageIdx = 0;

            // Loop over the pages
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
               sealedPage.VerifyChecksumIfEnabled().ThrowOnError();
               R__ASSERT(onDiskPage && (onDiskPage->GetSize() == sealedPage.GetBufferSize()));

               // Change compression if needed
               if (needsCompressionChange) {
                  auto taskFunc = [ // values in
                                     pageIdx, colRangeCompressionSettings, pageBufferBaseIdx, checksumSize,
                                     // const refs in
                                     &colElement, &pageInfo, &options,
                                     // refs in-out
                                     &sealedPage, &sealedPageBuffers]() {
                     // Step 1: prepare the source data.
                     // Unzip the source buffer into the zip staging buffer. This is a memcpy if the source was
                     // already uncompressed.
                     // Note that the checksum, if present, is not zipped, so we only need to unzip
                     // `sealedPage.GetDataSize()` bytes.
                     const auto uncompressedSize = colElement->GetSize() * sealedPage.GetNElements();
                     auto zipBuffer = std::make_unique<unsigned char[]>(uncompressedSize);
                     RNTupleDecompressor::Unzip(sealedPage.GetBuffer(), sealedPage.GetDataSize(), uncompressedSize,
                                                zipBuffer.get());

                     // Step 2: prepare the destination buffer.
                     if (uncompressedSize != sealedPage.GetDataSize()) {
                        // source page is compressed
                        R__ASSERT(colRangeCompressionSettings != 0);

                        // We need to reallocate sealedPage's buffer because we are going to recompress the data
                        // with a different algorithm/level. Since we don't know a priori how big that'll be, the
                        // only safe bet is to allocate a buffer big enough to hold as many bytes as the uncompressed
                        // data.
                        R__ASSERT(sealedPage.GetDataSize() < uncompressedSize);
                        auto &newBuf = sealedPageBuffers[pageBufferBaseIdx + pageIdx];
                        newBuf = std::make_unique<unsigned char[]>(uncompressedSize + checksumSize);
                        sealedPage.SetBuffer(newBuf.get());
                     } else {
                        // source page is uncompressed. We can reuse the sealedPage's buffer since it's big
                        // enough.
                        // Note that this does not necessarily mean that the column range's compressionSettings are 0,
                        // as a page might have been stored uncompressed because it was not compressible with its
                        // advertised compression settings.
                     }

                     const auto newNBytes =
                        RNTupleCompressor::Zip(zipBuffer.get(), uncompressedSize, options.fCompressionSettings,
                                               const_cast<void *>(sealedPage.GetBuffer()));
                     sealedPage.SetBufferSize(newNBytes + checksumSize);
                     if (pageInfo.fHasChecksum) {
                        // Calculate new checksum (this must happen after setting the new buffer size!)
                        sealedPage.ChecksumIfEnabled();
                     }
                  };

                  if (taskGroup)
                     taskGroup->Run(taskFunc);
                  else
                     taskFunc();
               }

               ++pageIdx;

            } // end of loop over pages

            if (taskGroup)
               taskGroup->Wait();
            sealedPagesV.push_back(std::move(sealedPages));
            sealedPageGroups.emplace_back(column.fColumnOutputId, sealedPagesV.back().cbegin(),
                                          sealedPagesV.back().cend());

         } // end of loop over srcColumns

         // Now commit all pages to the output
         destination.CommitSealedPageV(sealedPageGroups);

         // Commit the clusters
         destination.CommitCluster(clusterDesc.GetNEntries());
         nDstEntries += clusterDesc.GetNEntries();

         // Go to the next cluster
         clusterId = descriptor->FindNextClusterId(clusterId);

      } // end of loop over clusters

      // Commit all clusters for this input
      destination.CommitClusterGroup();

   } // end of loop over sources

   // Commit the output
   destination.CommitDataset();
}
