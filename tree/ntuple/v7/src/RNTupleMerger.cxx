/// \file RNTupleMerger.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>, Max Orok <maxwellorok@gmail.com>, Alaettin Serhan Mete <amete@anl.gov>,
/// Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2020-07-08
/// \warning This is part of the ROOT 7 prototype! It will
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
#include <ROOT/RColumnElementBase.hxx>
#include <TROOT.h>
#include <TFileMergeInfo.h>
#include <TError.h>
#include <TFile.h>
#include <TKey.h>

#include <algorithm>
#include <deque>
#include <inttypes.h> // for PRIu64
#include <initializer_list>
#include <unordered_map>
#include <vector>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Internal;

// TFile options parsing
// -------------------------------------------------------------------------------------
static bool BeginsWithDelimitedWord(const TString &str, const char *word)
{
   const Ssiz_t wordLen = strlen(word);
   if (str.Length() < wordLen)
      return false;
   if (!str.BeginsWith(word, TString::ECaseCompare::kIgnoreCase))
      return false;
   return str.Length() == wordLen || str(wordLen) == ' ';
}

template <typename T>
static std::optional<T> ParseStringOption(const TString &opts, const char *pattern,
                                          std::initializer_list<std::pair<const char *, T>> validValues)
{
   const Ssiz_t patternLen = strlen(pattern);
   assert(pattern[patternLen - 1] == '='); // we want to parse options with the format `option=Value`
   if (auto idx = opts.Index(pattern, 0, TString::ECaseCompare::kIgnoreCase);
       idx >= 0 && opts.Length() > idx + patternLen) {
      auto sub = TString(opts(idx + patternLen, opts.Length() - idx - patternLen));
      for (const auto &[name, value] : validValues) {
         if (BeginsWithDelimitedWord(sub, name)) {
            return value;
         }
      }
   }
   return std::nullopt;
}

static std::optional<ENTupleMergingMode> ParseOptionMergingMode(const TString &opts)
{
   return ParseStringOption<ENTupleMergingMode>(opts, "rntuple.MergingMode=",
                                                {
                                                   {"Filter", ENTupleMergingMode::kFilter},
                                                   {"Union", ENTupleMergingMode::kUnion},
                                                   {"Strict", ENTupleMergingMode::kStrict},
                                                });
}

static std::optional<ENTupleMergeErrBehavior> ParseOptionErrBehavior(const TString &opts)
{
   return ParseStringOption<ENTupleMergeErrBehavior>(opts, "rntuple.ErrBehavior=",
                                                     {
                                                        {"Abort", ENTupleMergeErrBehavior::kAbort},
                                                        {"Skip", ENTupleMergeErrBehavior::kSkip},
                                                     });
}
// -------------------------------------------------------------------------------------

// Entry point for TFileMerger. Internally calls RNTupleMerger::Merge().
Long64_t ROOT::RNTuple::Merge(TCollection *inputs, TFileMergeInfo *mergeInfo)
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
   ROOT::RNTuple *outNTuple = nullptr;
   if (outKey) {
      outNTuple = outKey->ReadObject<ROOT::RNTuple>();
      if (!outNTuple) {
         Error("RNTuple::Merge", "Output file already has key, but not of type RNTuple!");
         return -1;
      }
      // In principle, we should already be working on the RNTuple object from the output file, but just continue with
      // pointer we just got.
   }

   const bool defaultComp = mergeInfo->fOptions.Contains("DefaultCompression");
   const bool firstSrcComp = mergeInfo->fOptions.Contains("FirstSrcCompression");
   const bool extraVerbose = mergeInfo->fOptions.Contains("rntuple.ExtraVerbose");
   if (defaultComp && firstSrcComp) {
      // this should never happen through hadd, but a user may call RNTuple::Merge() from custom code.
      Warning("RNTuple::Merge", "Passed both options \"DefaultCompression\" and \"FirstSrcCompression\": "
                                "only the latter will apply.");
   }
   std::optional<std::uint32_t> compression;
   if (firstSrcComp) {
      // user passed -ff or -fk: use the same compression as the first RNTuple we find in the sources.
      // (do nothing here, the compression will be fetched below)
   } else if (!defaultComp) {
      // compression was explicitly passed by the user: use it.
      compression = outFile->GetCompressionSettings();
   } else {
      // user passed no compression-related options: use default
      compression = RCompressionSetting::EDefaults::kUseGeneralPurpose;
      Info("RNTuple::Merge", "Using the default compression: %u", *compression);
   }

   // The remaining entries are the input files
   std::vector<std::unique_ptr<RPageSourceFile>> sources;
   std::vector<RPageSource *> sourcePtrs;

   while (const auto &pitr = itr()) {
      TFile *inFile = dynamic_cast<TFile *>(pitr);
      ROOT::RNTuple *anchor = inFile ? inFile->Get<ROOT::RNTuple>(ntupleName.c_str()) : nullptr;
      if (!anchor) {
         Error("RNTuple::Merge", "Failed to retrieve RNTuple anchor named '%s' from file '%s'", ntupleName.c_str(),
               inFile->GetName());
         return -1;
      }

      auto source = RPageSourceFile::CreateFromAnchor(*anchor);
      if (!compression) {
         // Get the compression of this RNTuple and use it as the output compression.
         // We currently assume all column ranges have the same compression, so we just peek at the first one.
         source->Attach();
         auto descriptor = source->GetSharedDescriptorGuard();
         auto clusterIter = descriptor->GetClusterIterable();
         auto firstCluster = clusterIter.begin();
         if (firstCluster == clusterIter.end()) {
            Error("RNTuple::Merge",
                  "Asked to use the first source's compression as the output compression, but the "
                  "first source (file '%s') has an empty RNTuple, therefore the output compression could not be "
                  "determined.",
                  inFile->GetName());
            return -1;
         }
         auto colRangeIter = (*firstCluster).GetColumnRangeIterable();
         auto firstColRange = colRangeIter.begin();
         if (firstColRange == colRangeIter.end()) {
            Error("RNTuple::Merge",
                  "Asked to use the first source's compression as the output compression, but the "
                  "first source (file '%s') has an empty RNTuple, therefore the output compression could not be "
                  "determined.",
                  inFile->GetName());
            return -1;
         }
         compression = (*firstColRange).fCompressionSettings.value();
         Info("RNTuple::Merge", "Using the first RNTuple's compression: %u", *compression);
      }
      sources.push_back(std::move(source));
   }

   RNTupleWriteOptions writeOpts;
   assert(compression);
   writeOpts.SetCompression(*compression);
   auto destination = std::make_unique<RPageSinkFile>(ntupleName, *outFile, writeOpts);

   // If we already have an existing RNTuple, copy over its descriptor to support incremental merging
   if (outNTuple) {
      auto outSource = RPageSourceFile::CreateFromAnchor(*outNTuple);
      outSource->Attach();
      auto desc = outSource->GetSharedDescriptorGuard();
      destination->InitFromDescriptor(desc.GetRef());
   }

   // Interface conversion
   sourcePtrs.reserve(sources.size());
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   // Now merge
   RNTupleMerger merger{std::move(destination)};
   RNTupleMergeOptions mergerOpts;
   mergerOpts.fCompressionSettings = *compression;
   mergerOpts.fExtraVerbose = extraVerbose;
   if (auto mergingMode = ParseOptionMergingMode(mergeInfo->fOptions)) {
      mergerOpts.fMergingMode = *mergingMode;
   }
   if (auto errBehavior = ParseOptionErrBehavior(mergeInfo->fOptions)) {
      mergerOpts.fErrBehavior = *errBehavior;
   }
   merger.Merge(sourcePtrs, mergerOpts).ThrowOnError();

   // Provide the caller with a merged anchor object (even though we've already
   // written it).
   *this = *outFile->Get<ROOT::RNTuple>(ntupleName.c_str());

   return 0;
} catch (const RException &ex) {
   Error("RNTuple::Merge", "Exception thrown while merging: %s", ex.what());
   return -1;
}

namespace {
// Functor used to change the compression of a page to `options.fCompressionSettings`.
struct RChangeCompressionFunc {
   const RColumnElementBase &fSrcColElement;
   const RColumnElementBase &fDstColElement;
   const RNTupleMergeOptions &fMergeOptions;

   RPageStorage::RSealedPage &fSealedPage;
   RPageAllocator &fPageAlloc;
   std::uint8_t *fBuffer;

   void operator()() const
   {
      auto page = RPageSource::UnsealPage(fSealedPage, fSrcColElement, fPageAlloc).Unwrap();
      RPageSink::RSealPageConfig sealConf;
      sealConf.fElement = &fDstColElement;
      sealConf.fPage = &page;
      sealConf.fBuffer = fBuffer;
      sealConf.fCompressionSetting = *fMergeOptions.fCompressionSettings;
      sealConf.fWriteChecksum = fSealedPage.GetHasChecksum();
      auto refSealedPage = RPageSink::SealPage(sealConf);
      fSealedPage = refSealedPage;
   }
};

struct RCommonField {
   const RFieldDescriptor *fSrc;
   const RFieldDescriptor *fDst;

   RCommonField(const RFieldDescriptor *src, const RFieldDescriptor *dst) : fSrc(src), fDst(dst) {}
};

struct RDescriptorsComparison {
   std::vector<const RFieldDescriptor *> fExtraDstFields;
   std::vector<const RFieldDescriptor *> fExtraSrcFields;
   std::vector<RCommonField> fCommonFields;
};

struct RColumnOutInfo {
   DescriptorId_t fColumnId;
   ENTupleColumnType fColumnType;
};

// { fully.qualified.fieldName.colInputId => colOutputInfo }
using ColumnIdMap_t = std::unordered_map<std::string, RColumnOutInfo>;

struct RColumnInfoGroup {
   std::vector<RColumnMergeInfo> fExtraDstColumns;
   std::vector<RColumnMergeInfo> fCommonColumns;
};

} // namespace

// These structs cannot be in the anon namespace becase they're used in RNTupleMerger's private interface.
namespace ROOT::Experimental::Internal {
struct RColumnMergeInfo {
   // This column name is built as a dot-separated concatenation of the ancestry of
   // the columns' parent fields' names plus the index of the column itself.
   // e.g. "Muon.pt.x._0"
   std::string fColumnName;
   DescriptorId_t fInputId;
   DescriptorId_t fOutputId;
   ENTupleColumnType fColumnType;
   // If nullopt, use the default in-memory type
   std::optional<std::type_index> fInMemoryType;
   const RFieldDescriptor *fParentField;
};

// Data related to a single call of RNTupleMerger::Merge()
struct RNTupleMergeData {
   std::span<RPageSource *> fSources;
   RPageSink &fDestination;
   const RNTupleMergeOptions &fMergeOpts;
   const RNTupleDescriptor &fDstDescriptor;
   const RNTupleDescriptor *fSrcDescriptor = nullptr;

   std::vector<RColumnMergeInfo> fColumns;
   ColumnIdMap_t fColumnIdMap;

   NTupleSize_t fNumDstEntries = 0;

   RNTupleMergeData(std::span<RPageSource *> sources, RPageSink &destination, const RNTupleMergeOptions &mergeOpts)
      : fSources{sources}, fDestination{destination}, fMergeOpts{mergeOpts}, fDstDescriptor{destination.GetDescriptor()}
   {
   }
};

struct RSealedPageMergeData {
   // We use a std::deque so that references to the contained SealedPageSequence_t, and its iterators, are
   // never invalidated.
   std::deque<RPageStorage::SealedPageSequence_t> fPagesV;
   std::vector<RPageStorage::RSealedPageGroup> fGroups;
   std::vector<std::unique_ptr<std::uint8_t[]>> fBuffers;
};

std::ostream &operator<<(std::ostream &os, const std::optional<RColumnDescriptor::RValueRange> &x)
{
   if (x) {
      os << '(' << x->fMin << ", " << x->fMax << ')';
   } else {
      os << "(null)";
   }
   return os;
}

} // namespace ROOT::Experimental::Internal

static bool IsSplitOrUnsplitVersionOf(ENTupleColumnType a, ENTupleColumnType b)
{
   // clang-format off
   if (a == ENTupleColumnType::kInt16 && b == ENTupleColumnType::kSplitInt16) return true;
   if (a == ENTupleColumnType::kSplitInt16 && b == ENTupleColumnType::kInt16) return true;
   if (a == ENTupleColumnType::kInt32 && b == ENTupleColumnType::kSplitInt32) return true;
   if (a == ENTupleColumnType::kSplitInt32 && b == ENTupleColumnType::kInt32) return true;
   if (a == ENTupleColumnType::kInt64 && b == ENTupleColumnType::kSplitInt64) return true;
   if (a == ENTupleColumnType::kSplitInt64 && b == ENTupleColumnType::kInt64) return true;
   if (a == ENTupleColumnType::kUInt16 && b == ENTupleColumnType::kSplitUInt16) return true;
   if (a == ENTupleColumnType::kSplitUInt16 && b == ENTupleColumnType::kUInt16) return true;
   if (a == ENTupleColumnType::kUInt32 && b == ENTupleColumnType::kSplitUInt32) return true;
   if (a == ENTupleColumnType::kSplitUInt32 && b == ENTupleColumnType::kUInt32) return true;
   if (a == ENTupleColumnType::kUInt64 && b == ENTupleColumnType::kSplitUInt64) return true;
   if (a == ENTupleColumnType::kSplitUInt64 && b == ENTupleColumnType::kUInt64) return true;
   if (a == ENTupleColumnType::kIndex32 && b == ENTupleColumnType::kSplitIndex32) return true;
   if (a == ENTupleColumnType::kSplitIndex32 && b == ENTupleColumnType::kIndex32) return true;
   if (a == ENTupleColumnType::kIndex64 && b == ENTupleColumnType::kSplitIndex64) return true;
   if (a == ENTupleColumnType::kSplitIndex64 && b == ENTupleColumnType::kIndex64) return true;
   if (a == ENTupleColumnType::kReal32 && b == ENTupleColumnType::kSplitReal32) return true;
   if (a == ENTupleColumnType::kSplitReal32 && b == ENTupleColumnType::kReal32) return true;
   if (a == ENTupleColumnType::kReal64 && b == ENTupleColumnType::kSplitReal64) return true;
   if (a == ENTupleColumnType::kSplitReal64 && b == ENTupleColumnType::kReal64) return true;
   // clang-format on
   return false;
}

/// Compares the top level fields of `dst` and `src` and determines whether they can be merged or not.
/// In addition, returns the differences between `dst` and `src`'s structures
static ROOT::RResult<RDescriptorsComparison>
CompareDescriptorStructure(const RNTupleDescriptor &dst, const RNTupleDescriptor &src)
{
   // Cases:
   // 1. dst == src
   // 2. dst has fields that src hasn't
   // 3. src has fields that dst hasn't
   // 4. dst and src have fields that differ (compatible or incompatible)

   std::vector<std::string> errors;
   RDescriptorsComparison res;

   std::vector<RCommonField> commonFields;

   for (const auto &dstField : dst.GetTopLevelFields()) {
      const auto srcFieldId = src.FindFieldId(dstField.GetFieldName());
      if (srcFieldId != kInvalidDescriptorId) {
         const auto &srcField = src.GetFieldDescriptor(srcFieldId);
         commonFields.push_back({&srcField, &dstField});
      } else {
         res.fExtraDstFields.emplace_back(&dstField);
      }
   }
   for (const auto &srcField : src.GetTopLevelFields()) {
      const auto dstFieldId = dst.FindFieldId(srcField.GetFieldName());
      if (dstFieldId == kInvalidDescriptorId)
         res.fExtraSrcFields.push_back(&srcField);
   }

   // Check compatibility of common fields
   for (const auto &field : commonFields) {
      // NOTE: field.fSrc and field.fDst have the same name by construction
      const auto &fieldName = field.fSrc->GetFieldName();

      // Require that fields are both projected or both not projected
      bool projCompatible = field.fSrc->IsProjectedField() == field.fDst->IsProjectedField();
      if (!projCompatible) {
         std::stringstream ss;
         ss << "Field `" << fieldName << "` is incompatible with previously-seen field with that name because the "
            << (field.fSrc->IsProjectedField() ? "new" : "old") << " one is projected and the other isn't";
         errors.push_back(ss.str());
      } else if (field.fSrc->IsProjectedField()) {
         // if both fields are projected, verify that they point to the same real field
         const auto srcName = src.GetQualifiedFieldName(field.fSrc->GetProjectionSourceId());
         const auto dstName = dst.GetQualifiedFieldName(field.fDst->GetProjectionSourceId());
         if (srcName != dstName) {
            std::stringstream ss;
            ss << "Field `" << fieldName
               << "` is projected to a different field than a previously-seen field with the same name (old: "
               << dstName << ", new: " << srcName << ")";
            errors.push_back(ss.str());
         }
      }

      // Require that fields types match
      // TODO(gparolini): allow non-identical but compatible types
      const auto &srcTyName = field.fSrc->GetTypeName();
      const auto &dstTyName = field.fDst->GetTypeName();
      if (srcTyName != dstTyName) {
         std::stringstream ss;
         ss << "Field `" << fieldName
            << "` has a type incompatible with a previously-seen field with the same name: (old: " << dstTyName
            << ", new: " << srcTyName << ")";
         errors.push_back(ss.str());
      }

      // Require that type checksums match
      const auto srcTyChk = field.fSrc->GetTypeChecksum();
      const auto dstTyChk = field.fDst->GetTypeChecksum();
      if (srcTyChk && dstTyChk && *srcTyChk != *dstTyChk) {
         std::stringstream ss;
         ss << "Field `" << field.fSrc->GetFieldName()
            << "` has a different type checksum than previously-seen field with the same name";
         errors.push_back(ss.str());
      }

      // Require that type versions match
      const auto srcTyVer = field.fSrc->GetTypeVersion();
      const auto dstTyVer = field.fDst->GetTypeVersion();
      if (srcTyVer != dstTyVer) {
         std::stringstream ss;
         ss << "Field `" << field.fSrc->GetFieldName()
            << "` has a different type version than previously-seen field with the same name (old: " << dstTyVer
            << ", new: " << srcTyVer << ")";
         errors.push_back(ss.str());
      }

      // Require that column representations match
      const auto srcNCols = field.fSrc->GetLogicalColumnIds().size();
      const auto dstNCols = field.fDst->GetLogicalColumnIds().size();
      if (srcNCols != dstNCols) {
         std::stringstream ss;
         ss << "Field `" << field.fSrc->GetFieldName()
            << "` has a different number of columns than previously-seen field with the same name (old: " << dstNCols
            << ", new: " << srcNCols << ")";
         errors.push_back(ss.str());
      } else {
         for (auto i = 0u; i < srcNCols; ++i) {
            const auto srcColId = field.fSrc->GetLogicalColumnIds()[i];
            const auto dstColId = field.fDst->GetLogicalColumnIds()[i];
            const auto &srcCol = src.GetColumnDescriptor(srcColId);
            const auto &dstCol = dst.GetColumnDescriptor(dstColId);
            // TODO(gparolini): currently we refuse to merge columns of different types unless they are Split/non-Split
            // version of the same type, because we know how to treat that specific case. We should also properly handle
            // different but compatible types.
            if (srcCol.GetType() != dstCol.GetType() &&
                !IsSplitOrUnsplitVersionOf(srcCol.GetType(), dstCol.GetType())) {
               std::stringstream ss;
               ss << i << "-th column of field `" << field.fSrc->GetFieldName()
                  << "` has a different column type of the same column on the previously-seen field with the same name "
                     "(old: "
                  << RColumnElementBase::GetColumnTypeName(srcCol.GetType())
                  << ", new: " << RColumnElementBase::GetColumnTypeName(dstCol.GetType()) << ")";
               errors.push_back(ss.str());
            }
            if (srcCol.GetBitsOnStorage() != dstCol.GetBitsOnStorage()) {
               std::stringstream ss;
               ss << i << "-th column of field `" << field.fSrc->GetFieldName()
                  << "` has a different number of bits of the same column on the previously-seen field with the same "
                     "name "
                     "(old: "
                  << srcCol.GetBitsOnStorage() << ", new: " << dstCol.GetBitsOnStorage() << ")";
               errors.push_back(ss.str());
            }
            if (srcCol.GetValueRange() != dstCol.GetValueRange()) {
               std::stringstream ss;
               ss << i << "-th column of field `" << field.fSrc->GetFieldName()
                  << "` has a different value range of the same column on the previously-seen field with the same name "
                     "(old: "
                  << srcCol.GetValueRange() << ", new: " << dstCol.GetValueRange() << ")";
               errors.push_back(ss.str());
            }
            if (srcCol.GetRepresentationIndex() > 0) {
               std::stringstream ss;
               ss << i << "-th column of field `" << field.fSrc->GetFieldName()
                  << "` has a representation index higher than 0. This is not supported yet by the merger.";
               errors.push_back(ss.str());
            }
         }
      }
   }

   std::string errMsg;
   for (const auto &err : errors)
      errMsg += std::string("\n  * ") + err;

   if (!errMsg.empty())
      errMsg = errMsg.substr(1); // strip initial newline

   if (errMsg.length())
      return R__FAIL(errMsg);

   res.fCommonFields.reserve(commonFields.size());
   for (const auto &[srcField, dstField] : commonFields) {
      res.fCommonFields.emplace_back(srcField, dstField);
   }

   // TODO(gparolini): we should exhaustively check the field tree rather than just the top level fields,
   // in case the user forgets to change the version number on one field.

   return ROOT::RResult(res);
}

// Applies late model extension to `destination`, adding all `newFields` to it.
static void ExtendDestinationModel(std::span<const RFieldDescriptor *> newFields, RNTupleModel &dstModel,
                                   RNTupleMergeData &mergeData, std::vector<RCommonField> &commonFields)
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
   Info("RNTuple::Merge", "%s: adding %s to the destination model (entry #%" PRIu64 ").", msg.c_str(),
        (newFields.size() > 1 ? "them" : "it"), mergeData.fNumDstEntries);

   changeset.fAddedFields.reserve(newFields.size());
   for (const auto *fieldDesc : newFields) {
      auto field = fieldDesc->CreateField(*mergeData.fSrcDescriptor);
      if (fieldDesc->IsProjectedField())
         changeset.fAddedProjectedFields.emplace_back(field.get());
      else
         changeset.fAddedFields.emplace_back(field.get());
      changeset.fModel.AddField(std::move(field));
   }
   dstModel.Freeze();
   mergeData.fDestination.UpdateSchema(changeset, mergeData.fNumDstEntries);

   commonFields.reserve(commonFields.size() + newFields.size());
   for (const auto *field : newFields) {
      const auto newFieldInDstId = mergeData.fDstDescriptor.FindFieldId(field->GetFieldName());
      const auto &newFieldInDst = mergeData.fDstDescriptor.GetFieldDescriptor(newFieldInDstId);
      commonFields.emplace_back(field, &newFieldInDst);
   }
}

// Merges all columns appearing both in the source and destination RNTuples, just copying them if their
// compression matches ("fast merge") or by unsealing and resealing them with the proper compression.
void RNTupleMerger::MergeCommonColumns(RClusterPool &clusterPool, DescriptorId_t clusterId,
                                       std::span<RColumnMergeInfo> commonColumns,
                                       const RCluster::ColumnSet_t &commonColumnSet,
                                       RSealedPageMergeData &sealedPageData, const RNTupleMergeData &mergeData)
{
   assert(commonColumns.size() == commonColumnSet.size());
   if (commonColumns.empty())
      return;

   const RCluster *cluster = clusterPool.GetCluster(clusterId, commonColumnSet);
   // we expect the cluster pool to contain the requested set of columns, since they were
   // validated by CompareDescriptorStructure().
   assert(cluster);

   const auto &clusterDesc = mergeData.fSrcDescriptor->GetClusterDescriptor(clusterId);

   for (const auto &column : commonColumns) {
      const auto &columnId = column.fInputId;
      R__ASSERT(clusterDesc.ContainsColumn(columnId));

      const auto &columnDesc = mergeData.fSrcDescriptor->GetColumnDescriptor(columnId);
      const auto srcColElement = column.fInMemoryType
                                    ? GenerateColumnElement(*column.fInMemoryType, columnDesc.GetType())
                                    : RColumnElementBase::Generate(columnDesc.GetType());
      const auto dstColElement = column.fInMemoryType ? GenerateColumnElement(*column.fInMemoryType, column.fColumnType)
                                                      : RColumnElementBase::Generate(column.fColumnType);

      // Now get the pages for this column in this cluster
      const auto &pages = clusterDesc.GetPageRange(columnId);

      RPageStorage::SealedPageSequence_t sealedPages;
      sealedPages.resize(pages.fPageInfos.size());

      // Each column range potentially has a distinct compression settings
      const auto colRangeCompressionSettings = clusterDesc.GetColumnRange(columnId).fCompressionSettings.value();
      const bool needsCompressionChange =
         colRangeCompressionSettings != mergeData.fMergeOpts.fCompressionSettings.value();
      if (needsCompressionChange && mergeData.fMergeOpts.fExtraVerbose)
         Info("RNTuple::Merge", "Column %s: changing source compression from %d to %d", column.fColumnName.c_str(),
              colRangeCompressionSettings, mergeData.fMergeOpts.fCompressionSettings.value());

      size_t pageBufferBaseIdx = sealedPageData.fBuffers.size();
      // If the column range already has the right compression we don't need to allocate any new buffer, so we don't
      // bother reserving memory for them.
      if (needsCompressionChange)
         sealedPageData.fBuffers.resize(sealedPageData.fBuffers.size() + pages.fPageInfos.size());

      // Loop over the pages
      std::uint64_t pageIdx = 0;
      for (const auto &pageInfo : pages.fPageInfos) {
         assert(pageIdx < sealedPages.size());
         assert(sealedPageData.fBuffers.size() == 0 || pageIdx < sealedPageData.fBuffers.size());

         ROnDiskPage::Key key{columnId, pageIdx};
         auto onDiskPage = cluster->GetOnDiskPage(key);

         const auto checksumSize = pageInfo.fHasChecksum * RPageStorage::kNBytesPageChecksum;
         RPageStorage::RSealedPage &sealedPage = sealedPages[pageIdx];
         sealedPage.SetNElements(pageInfo.fNElements);
         sealedPage.SetHasChecksum(pageInfo.fHasChecksum);
         sealedPage.SetBufferSize(pageInfo.fLocator.GetNBytesOnStorage() + checksumSize);
         sealedPage.SetBuffer(onDiskPage->GetAddress());
         // TODO(gparolini): more graceful error handling (skip the page?)
         sealedPage.VerifyChecksumIfEnabled().ThrowOnError();
         R__ASSERT(onDiskPage && (onDiskPage->GetSize() == sealedPage.GetBufferSize()));

         if (needsCompressionChange) {
            const auto uncompressedSize = srcColElement->GetSize() * sealedPage.GetNElements();
            auto &buffer = sealedPageData.fBuffers[pageBufferBaseIdx + pageIdx];
            buffer = MakeUninitArray<std::uint8_t>(uncompressedSize + checksumSize);
            RChangeCompressionFunc compressTask{
               *srcColElement, *dstColElement, mergeData.fMergeOpts, sealedPage, *fPageAlloc, buffer.get(),
            };

            if (fTaskGroup)
               fTaskGroup->Run(compressTask);
            else
               compressTask();
         }

         ++pageIdx;

      } // end of loop over pages

      if (fTaskGroup)
         fTaskGroup->Wait();

      sealedPageData.fPagesV.push_back(std::move(sealedPages));
      sealedPageData.fGroups.emplace_back(column.fOutputId, sealedPageData.fPagesV.back().cbegin(),
                                          sealedPageData.fPagesV.back().cend());
   } // end loop over common columns
}

// Generates default values for columns that are not present in the current source RNTuple
// but are present in the destination's schema.
static void GenerateExtraDstColumns(size_t nClusterEntries, std::span<RColumnMergeInfo> extraDstColumns,
                                    RSealedPageMergeData &sealedPageData, const RNTupleMergeData &mergeData)
{
   for (const auto &column : extraDstColumns) {
      const auto &columnId = column.fInputId;
      const auto &columnDesc = mergeData.fDstDescriptor.GetColumnDescriptor(columnId);
      const RFieldDescriptor *field = column.fParentField;

      // Skip all auxiliary columns
      if (field->GetLogicalColumnIds()[0] != columnId)
         continue;

      // Check if this column is a child of a Collection or a Variant. If so, it has no data
      // and can be skipped.
      bool skipColumn = false;
      auto nRepetitions = std::max<std::uint64_t>(field->GetNRepetitions(), 1);
      for (auto parentId = field->GetParentId(); parentId != kInvalidDescriptorId;) {
         const RFieldDescriptor &parent = mergeData.fSrcDescriptor->GetFieldDescriptor(parentId);
         if (parent.GetStructure() == ENTupleStructure::kCollection ||
             parent.GetStructure() == ENTupleStructure::kVariant) {
            skipColumn = true;
            break;
         }
         nRepetitions *= std::max<std::uint64_t>(parent.GetNRepetitions(), 1);
         parentId = parent.GetParentId();
      }
      if (skipColumn)
         continue;

      const auto structure = field->GetStructure();

      if (structure == ENTupleStructure::kStreamer) {
         Fatal(
            "RNTuple::Merge",
            "Destination RNTuple contains a streamer field (%s) that is not present in one of the sources. "
            "Creating a default value for a streamer field is ill-defined, therefore the merging process will abort.",
            field->GetFieldName().c_str());
         continue;
      }

      // NOTE: we cannot have a Record here because it has no associated columns.
      R__ASSERT(structure == ENTupleStructure::kCollection || structure == ENTupleStructure::kVariant ||
                structure == ENTupleStructure::kLeaf);

      const auto colElement = RColumnElementBase::Generate(columnDesc.GetType());
      const auto nElements = nClusterEntries * nRepetitions;
      const auto nBytesOnStorage = colElement->GetPackedSize(nElements);
      constexpr auto kPageSizeLimit = 256 * 1024;
      // TODO(gparolini): consider coalescing the last page if its size is less than some threshold
      const size_t nPages = nBytesOnStorage / kPageSizeLimit + !!(nBytesOnStorage % kPageSizeLimit);
      for (size_t i = 0; i < nPages; ++i) {
         const auto pageSize = (i < nPages - 1) ? kPageSizeLimit : nBytesOnStorage - kPageSizeLimit * (nPages - 1);
         const auto checksumSize = RPageStorage::kNBytesPageChecksum;
         const auto bufSize = pageSize + checksumSize;
         auto &buffer = sealedPageData.fBuffers.emplace_back(new unsigned char[bufSize]);

         RPageStorage::RSealedPage sealedPage{buffer.get(), bufSize, static_cast<std::uint32_t>(nElements), true};
         memset(buffer.get(), 0, pageSize);
         sealedPage.ChecksumIfEnabled();

         sealedPageData.fPagesV.push_back({sealedPage});
      }

      sealedPageData.fGroups.emplace_back(column.fOutputId, sealedPageData.fPagesV.back().cbegin(),
                                          sealedPageData.fPagesV.back().cend());
   }
}

// Iterates over all clusters of `source` and merges their pages into `destination`.
// It is assumed that all columns in `commonColumns` are present (and compatible) in both the source and
// the destination's schemas.
// The pages may be "fast-merged" (i.e. simply copied with no decompression/recompression) if the target
// compression is unspecified or matches the original compression settings.
void RNTupleMerger::MergeSourceClusters(RPageSource &source, std::span<RColumnMergeInfo> commonColumns,
                                        std::span<RColumnMergeInfo> extraDstColumns, RNTupleMergeData &mergeData)
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
   DescriptorId_t clusterId = mergeData.fSrcDescriptor->FindClusterId(0, 0);
   while (clusterId != kInvalidDescriptorId) {
      const auto &clusterDesc = mergeData.fSrcDescriptor->GetClusterDescriptor(clusterId);
      const auto nClusterEntries = clusterDesc.GetNEntries();
      R__ASSERT(nClusterEntries > 0);

      RSealedPageMergeData sealedPageData;

      if (!commonColumnSet.empty()) {
         MergeCommonColumns(clusterPool, clusterId, commonColumns, commonColumnSet, sealedPageData, mergeData);
      }

      if (!extraDstColumnSet.empty()) {
         GenerateExtraDstColumns(nClusterEntries, extraDstColumns, sealedPageData, mergeData);
      }

      // Commit the pages and the clusters
      mergeData.fDestination.CommitSealedPageV(sealedPageData.fGroups);
      mergeData.fDestination.CommitCluster(nClusterEntries);
      mergeData.fNumDstEntries += nClusterEntries;

      // Go to the next cluster
      clusterId = mergeData.fSrcDescriptor->FindNextClusterId(clusterId);
   }

   // TODO(gparolini): when we get serious about huge file support (>~ 100GB) we might want to check here
   // the size of the running page list and commit a cluster group when it exceeds some threshold,
   // which would prevent the page list from getting too large.
   // However, as of today, we aren't really handling such huge files, and even relatively big ones
   // such as the CMS dataset have a page list size of about only 2 MB.
   // So currently we simply merge all cluster groups into one.
}

static std::optional<std::type_index> ColumnInMemoryType(std::string_view fieldType, ENTupleColumnType onDiskType)
{
   if (onDiskType == ENTupleColumnType::kIndex32 || onDiskType == ENTupleColumnType::kSplitIndex32 ||
       onDiskType == ENTupleColumnType::kIndex64 || onDiskType == ENTupleColumnType::kSplitIndex64)
      return typeid(ROOT::Experimental::Internal::RColumnIndex);

   if (onDiskType == ENTupleColumnType::kSwitch)
      return typeid(ROOT::Experimental::Internal::RColumnSwitch);

   if (fieldType == "bool") {
      return typeid(bool);
   } else if (fieldType == "std::byte") {
      return typeid(std::byte);
   } else if (fieldType == "char") {
      return typeid(char);
   } else if (fieldType == "std::int8_t") {
      return typeid(std::int8_t);
   } else if (fieldType == "std::uint8_t") {
      return typeid(std::uint8_t);
   } else if (fieldType == "std::int16_t") {
      return typeid(std::int16_t);
   } else if (fieldType == "std::uint16_t") {
      return typeid(std::uint16_t);
   } else if (fieldType == "std::int32_t") {
      return typeid(std::int32_t);
   } else if (fieldType == "std::uint32_t") {
      return typeid(std::uint32_t);
   } else if (fieldType == "std::int64_t") {
      return typeid(std::int64_t);
   } else if (fieldType == "std::uint64_t") {
      return typeid(std::uint64_t);
   } else if (fieldType == "float") {
      return typeid(float);
   } else if (fieldType == "double") {
      return typeid(double);
   }

   // if the type is not one of those above, we use the default in-memory type.
   return std::nullopt;
}

// Given a field, fill `columns` and `colIdMap` with information about all columns belonging to it and its subfields.
// `colIdMap` is used to map matching columns from different sources to the same output column in the destination.
// We match columns by their "fully qualified name", which is the concatenation of their ancestor fields' names
// and the column index.
// By this point, since we called `CompareDescriptorStructure()` earlier, we should be guaranteed that two matching
// columns will have at least compatible representations.
// NOTE: srcFieldDesc and dstFieldDesc may alias.
static void AddColumnsFromField(std::vector<RColumnMergeInfo> &columns, const RNTupleDescriptor &srcDesc,
                                RNTupleMergeData &mergeData, const RFieldDescriptor &srcFieldDesc,
                                const RFieldDescriptor &dstFieldDesc, const std::string &prefix = "")
{
   std::string name = prefix + '.' + srcFieldDesc.GetFieldName();

   const auto &columnIds = srcFieldDesc.GetLogicalColumnIds();
   columns.reserve(columns.size() + columnIds.size());
   // NOTE: here we can match the src and dst columns by column index because we forbid merging fields with
   // different column representations.
   for (auto i = 0u; i < srcFieldDesc.GetLogicalColumnIds().size(); ++i) {
      // We don't want to try and merge alias columns
      if (srcFieldDesc.IsProjectedField())
         continue;

      auto srcColumnId = srcFieldDesc.GetLogicalColumnIds()[i];
      const auto &srcColumn = srcDesc.GetColumnDescriptor(srcColumnId);
      RColumnMergeInfo info{};
      info.fColumnName = name + '.' + std::to_string(srcColumn.GetIndex());
      info.fInputId = srcColumn.GetPhysicalId();
      // Since the parent field is only relevant for extra dst columns, the choice of src or dstFieldDesc as a parent
      // is arbitrary (they're the same field).
      info.fParentField = &dstFieldDesc;

      if (auto it = mergeData.fColumnIdMap.find(info.fColumnName); it != mergeData.fColumnIdMap.end()) {
         info.fOutputId = it->second.fColumnId;
         info.fColumnType = it->second.fColumnType;
      } else {
         info.fOutputId = mergeData.fColumnIdMap.size();
         // NOTE(gparolini): map the type of src column to the type of dst column.
         // This mapping is only relevant for common columns and it's done to ensure we keep a consistent
         // on-disk representation of the same column.
         // This is also important to do for first source when it is used to generate the destination sink,
         // because even in that case their column representations may differ.
         // e.g. if the destination has a different compression than the source, an integer column might be
         // zigzag-encoded in the source but not in the destination.
         auto dstColumnId = dstFieldDesc.GetLogicalColumnIds()[i];
         const auto &dstColumn = mergeData.fDstDescriptor.GetColumnDescriptor(dstColumnId);
         info.fColumnType = dstColumn.GetType();
         mergeData.fColumnIdMap[info.fColumnName] = {info.fOutputId, info.fColumnType};
      }

      if (mergeData.fMergeOpts.fExtraVerbose) {
         Info("RNTuple::Merge",
              "Adding column %s with log.id %" PRIu64 ", phys.id %" PRIu64 ", type %s "
              " -> log.id %" PRIu64 ", type %s",
              info.fColumnName.c_str(), srcColumnId, srcColumn.GetPhysicalId(),
              RColumnElementBase::GetColumnTypeName(srcColumn.GetType()), info.fOutputId,
              RColumnElementBase::GetColumnTypeName(info.fColumnType));
      }

      // Since we disallow merging fields of different types, src and dstFieldDesc must have the same type name.
      assert(srcFieldDesc.GetTypeName() == dstFieldDesc.GetTypeName());
      info.fInMemoryType = ColumnInMemoryType(srcFieldDesc.GetTypeName(), info.fColumnType);
      columns.emplace_back(info);
   }

   const auto &srcChildrenIds = srcFieldDesc.GetLinkIds();
   const auto &dstChildrenIds = dstFieldDesc.GetLinkIds();
   assert(srcChildrenIds.size() == dstChildrenIds.size());
   for (auto i = 0u; i < srcChildrenIds.size(); ++i) {
      const auto &srcChild = srcDesc.GetFieldDescriptor(srcChildrenIds[i]);
      const auto &dstChild = mergeData.fDstDescriptor.GetFieldDescriptor(dstChildrenIds[i]);
      AddColumnsFromField(columns, srcDesc, mergeData, srcChild, dstChild, name);
   }
}

// Converts the fields comparison data to the corresponding column information.
// While doing so, it collects such information in `colIdMap`, which is used by later calls to this function
// to map already-seen column names to their chosen outputId, type and so on.
static RColumnInfoGroup
GatherColumnInfos(const RDescriptorsComparison &descCmp, const RNTupleDescriptor &srcDesc, RNTupleMergeData &mergeData)
{
   RColumnInfoGroup res;
   for (const RFieldDescriptor *field : descCmp.fExtraDstFields) {
      AddColumnsFromField(res.fExtraDstColumns, mergeData.fDstDescriptor, mergeData, *field, *field);
   }
   for (const auto &[srcField, dstField] : descCmp.fCommonFields) {
      AddColumnsFromField(res.fCommonColumns, srcDesc, mergeData, *srcField, *dstField);
   }
   return res;
}

RNTupleMerger::RNTupleMerger(std::unique_ptr<RPageSink> destination)
   // TODO(gparolini): consider using an arena allocator instead, since we know the precise lifetime
   // of the RNTuples we are going to handle (e.g. we can reset the arena at every source)
   : fDestination(std::move(destination)), fPageAlloc(std::make_unique<RPageAllocatorHeap>())
{
   R__ASSERT(fDestination);

#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled())
      fTaskGroup = TTaskGroup();
#endif
}

ROOT::RResult<void> RNTupleMerger::Merge(std::span<RPageSource *> sources, const RNTupleMergeOptions &mergeOptsIn)
{
   RNTupleMergeOptions mergeOpts = mergeOptsIn;

   assert(fDestination);

   // Set compression settings if unset and verify it's compatible with the sink
   {
      const auto dstCompSettings = fDestination->GetWriteOptions().GetCompression();
      if (!mergeOpts.fCompressionSettings) {
         mergeOpts.fCompressionSettings = dstCompSettings;
      } else if (*mergeOpts.fCompressionSettings != dstCompSettings) {
         return R__FAIL(std::string("The compression given to RNTupleMergeOptions is different from that of the "
                                    "sink! (opts: ") +
                        std::to_string(*mergeOpts.fCompressionSettings) + ", sink: " + std::to_string(dstCompSettings) +
                        ") This is currently unsupported.");
      }
   }

   RNTupleMergeData mergeData{sources, *fDestination, mergeOpts};

   std::unique_ptr<RNTupleModel> model; // used to initialize the schema of the output RNTuple

#define SKIP_OR_ABORT(errMsg)                                                        \
   do {                                                                              \
      if (mergeOpts.fErrBehavior == ENTupleMergeErrBehavior::kSkip) {                \
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
      mergeData.fSrcDescriptor = &srcDescriptor.GetRef();

      // Create sink from the input model if not initialized
      if (!fDestination->IsInitialized()) {
         auto opts = RNTupleDescriptor::RCreateModelOptions();
         opts.fReconstructProjections = true;
         model = srcDescriptor->CreateModel(opts);
         fDestination->Init(*model);
      }

      for (const auto &extraTypeInfoDesc : srcDescriptor->GetExtraTypeInfoIterable())
         fDestination->UpdateExtraTypeInfo(extraTypeInfoDesc);

      auto descCmpRes = CompareDescriptorStructure(mergeData.fDstDescriptor, srcDescriptor.GetRef());
      if (!descCmpRes) {
         SKIP_OR_ABORT(
            std::string("Source RNTuple will be skipped due to incompatible schema with the fDestination:\n") +
            descCmpRes.GetError()->GetReport());
      }
      auto descCmp = descCmpRes.Unwrap();

      // If the current source is missing some fields and we're not in Union mode, error
      // (if we are in Union mode, MergeSourceClusters will fill the missing fields with default values).
      if (mergeOpts.fMergingMode != ENTupleMergingMode::kUnion && !descCmp.fExtraDstFields.empty()) {
         std::string msg = "Source RNTuple is missing the following fields:";
         for (const auto *field : descCmp.fExtraDstFields) {
            msg += "\n  " + field->GetFieldName() + " : " + field->GetTypeName();
         }
         SKIP_OR_ABORT(msg);
      }

      // handle extra src fields
      if (descCmp.fExtraSrcFields.size()) {
         if (mergeOpts.fMergingMode == ENTupleMergingMode::kUnion) {
            // late model extension for all fExtraSrcFields in Union mode
            ExtendDestinationModel(descCmp.fExtraSrcFields, *model, mergeData, descCmp.fCommonFields);
         } else if (mergeOpts.fMergingMode == ENTupleMergingMode::kStrict) {
            // If the current source has extra fields and we're in Strict mode, error
            std::string msg = "Source RNTuple has extra fields that the fDestination RNTuple doesn't have:";
            for (const auto *field : descCmp.fExtraSrcFields) {
               msg += "\n  " + field->GetFieldName() + " : " + field->GetTypeName();
            }
            SKIP_OR_ABORT(msg);
         }
      }

      // handle extra dst fields & common fields
      auto columnInfos = GatherColumnInfos(descCmp, srcDescriptor.GetRef(), mergeData);
      MergeSourceClusters(*source, columnInfos.fCommonColumns, columnInfos.fExtraDstColumns, mergeData);
   } // end loop over sources

   // Commit the output
   fDestination->CommitClusterGroup();
   fDestination->CommitDataset();

   return RResult<void>::Success();
}
