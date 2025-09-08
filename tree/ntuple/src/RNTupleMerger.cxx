/// \file RNTupleMerger.cxx
/// \ingroup NTuple
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
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RClusterPool.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RColumnElementBase.hxx>
#include <TROOT.h>
#include <TFileMergeInfo.h>
#include <TFile.h>
#include <TKey.h>

#include <algorithm>
#include <deque>
#include <inttypes.h> // for PRIu64
#include <initializer_list>
#include <unordered_map>
#include <vector>

using ROOT::ENTupleColumnType;
using ROOT::RNTupleModel;
using ROOT::Internal::MakeUninitArray;
using ROOT::Internal::RCluster;
using ROOT::Internal::RColumnElementBase;
using ROOT::Internal::RNTupleSerializer;
using ROOT::Internal::RPageSink;
using ROOT::Internal::RPageSource;
using ROOT::Internal::RPageSourceFile;
using ROOT::Internal::RPageStorage;

using namespace ROOT::Experimental::Internal;

static ROOT::RLogChannel &NTupleMergeLog()
{
   static ROOT::RLogChannel sLog("ROOT.NTuple.Merge");
   return sLog;
}

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
      R__LOG_ERROR(NTupleMergeLog()) << "Invalid inputs.";
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
      R__LOG_ERROR(NTupleMergeLog()) << "Second input parameter should be a TFile, but it's a "
                                     << secondArg->ClassName() << ".";
      return -1;
   }

   // Check if the output file already has a key with that name
   TKey *outKey = outFile->FindKey(ntupleName.c_str());
   ROOT::RNTuple *outNTuple = nullptr;
   if (outKey) {
      outNTuple = outKey->ReadObject<ROOT::RNTuple>();
      if (!outNTuple) {
         R__LOG_ERROR(NTupleMergeLog()) << "Output file already has key, but not of type RNTuple!";
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
      R__LOG_WARNING(NTupleMergeLog()) << "Passed both options \"DefaultCompression\" and \"FirstSrcCompression\": "
                                          "only the latter will apply.";
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
      R__LOG_INFO(NTupleMergeLog()) << "Using the default compression: " << *compression;
   }

   // The remaining entries are the input files
   std::vector<std::unique_ptr<RPageSourceFile>> sources;
   std::vector<RPageSource *> sourcePtrs;

   while (const auto &pitr = itr()) {
      TFile *inFile = dynamic_cast<TFile *>(pitr);
      ROOT::RNTuple *anchor = inFile ? inFile->Get<ROOT::RNTuple>(ntupleName.c_str()) : nullptr;
      if (!anchor) {
         R__LOG_INFO(NTupleMergeLog()) << "No RNTuple anchor named '" << ntupleName << "' from file '"
                                       << inFile->GetName() << "'";
         continue;
      }

      auto source = RPageSourceFile::CreateFromAnchor(*anchor);
      if (!compression) {
         // Get the compression of this RNTuple and use it as the output compression.
         // We currently assume all column ranges have the same compression, so we just peek at the first one.
         source->Attach(RNTupleSerializer::EDescriptorDeserializeMode::kRaw);
         auto descriptor = source->GetSharedDescriptorGuard();
         auto clusterIter = descriptor->GetClusterIterable();
         auto firstCluster = clusterIter.begin();
         if (firstCluster == clusterIter.end()) {
            R__LOG_ERROR(NTupleMergeLog())
               << "Asked to use the first source's compression as the output compression, but the "
                  "first source (file '"
               << inFile->GetName()
               << "') has an empty RNTuple, therefore the output compression could not be "
                  "determined.";
            return -1;
         }
         auto colRangeIter = (*firstCluster).GetColumnRangeIterable();
         auto firstColRange = colRangeIter.begin();
         if (firstColRange == colRangeIter.end()) {
            R__LOG_ERROR(NTupleMergeLog())
               << "Asked to use the first source's compression as the output compression, but the "
                  "first source (file '"
               << inFile->GetName()
               << "') has an empty RNTuple, therefore the output compression could not be "
                  "determined.";
            return -1;
         }
         compression = (*firstColRange).GetCompressionSettings().value();
         R__LOG_INFO(NTupleMergeLog()) << "Using the first RNTuple's compression: " << *compression;
      }
      sources.push_back(std::move(source));
   }

   RNTupleWriteOptions writeOpts;
   assert(compression);
   writeOpts.SetCompression(*compression);
   auto destination = std::make_unique<ROOT::Internal::RPageSinkFile>(ntupleName, *outFile, writeOpts);
   std::unique_ptr<ROOT::RNTupleModel> model;
   // If we already have an existing RNTuple, copy over its descriptor to support incremental merging
   if (outNTuple) {
      auto outSource = RPageSourceFile::CreateFromAnchor(*outNTuple);
      outSource->Attach(RNTupleSerializer::EDescriptorDeserializeMode::kForWriting);
      auto desc = outSource->GetSharedDescriptorGuard();
      model = destination->InitFromDescriptor(desc.GetRef(), true /* copyClusters */);
   }

   // Interface conversion
   sourcePtrs.reserve(sources.size());
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   // Now merge
   RNTupleMerger merger{std::move(destination), std::move(model)};
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
} catch (const std::exception &ex) {
   R__LOG_ERROR(NTupleMergeLog()) << "Exception thrown while merging: " << ex.what();
   return -1;
}

namespace {
// Functor used to change the compression of a page to `options.fCompressionSettings`.
struct RChangeCompressionFunc {
   const RColumnElementBase &fSrcColElement;
   const RColumnElementBase &fDstColElement;
   const RNTupleMergeOptions &fMergeOptions;

   RPageStorage::RSealedPage &fSealedPage;
   ROOT::Internal::RPageAllocator &fPageAlloc;
   std::uint8_t *fBuffer;

   void operator()() const
   {
      auto page = RPageSource::UnsealPage(fSealedPage, fSrcColElement, fPageAlloc).Unwrap();
      RPageSink::RSealPageConfig sealConf;
      sealConf.fElement = &fDstColElement;
      sealConf.fPage = &page;
      sealConf.fBuffer = fBuffer;
      sealConf.fCompressionSettings = *fMergeOptions.fCompressionSettings;
      sealConf.fWriteChecksum = fSealedPage.GetHasChecksum();
      auto refSealedPage = RPageSink::SealPage(sealConf);
      fSealedPage = refSealedPage;
   }
};

struct RCommonField {
   const ROOT::RFieldDescriptor *fSrc;
   const ROOT::RFieldDescriptor *fDst;

   RCommonField(const ROOT::RFieldDescriptor &src, const ROOT::RFieldDescriptor &dst) : fSrc(&src), fDst(&dst) {}
};

struct RDescriptorsComparison {
   std::vector<const ROOT::RFieldDescriptor *> fExtraDstFields;
   std::vector<const ROOT::RFieldDescriptor *> fExtraSrcFields;
   std::vector<RCommonField> fCommonFields;
};

struct RColumnOutInfo {
   ROOT::DescriptorId_t fColumnId;
   ENTupleColumnType fColumnType;
};

// { fully.qualified.fieldName.colInputId => colOutputInfo }
using ColumnIdMap_t = std::unordered_map<std::string, RColumnOutInfo>;

struct RColumnInfoGroup {
   std::vector<RColumnMergeInfo> fExtraDstColumns;
   // These are sorted by InputId
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
   // The column id in the source RNTuple
   ROOT::DescriptorId_t fInputId;
   // The corresponding column id in the destination RNTuple (the mapping happens in AddColumnsFromField())
   ROOT::DescriptorId_t fOutputId;
   ENTupleColumnType fColumnType;
   // If nullopt, use the default in-memory type
   std::optional<std::type_index> fInMemoryType;
   const ROOT::RFieldDescriptor *fParentFieldDescriptor;
   const ROOT::RNTupleDescriptor *fParentNTupleDescriptor;
};

// Data related to a single call of RNTupleMerger::Merge()
struct RNTupleMergeData {
   std::span<RPageSource *> fSources;
   RPageSink &fDestination;
   const RNTupleMergeOptions &fMergeOpts;
   const ROOT::RNTupleDescriptor &fDstDescriptor;
   const ROOT::RNTupleDescriptor *fSrcDescriptor = nullptr;

   std::vector<RColumnMergeInfo> fColumns;
   ColumnIdMap_t fColumnIdMap;

   ROOT::NTupleSize_t fNumDstEntries = 0;

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

std::ostream &operator<<(std::ostream &os, const std::optional<ROOT::RColumnDescriptor::RValueRange> &x)
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
CompareDescriptorStructure(const ROOT::RNTupleDescriptor &dst, const ROOT::RNTupleDescriptor &src)
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
      if (srcFieldId != ROOT::kInvalidDescriptorId) {
         const auto &srcField = src.GetFieldDescriptor(srcFieldId);
         commonFields.push_back({srcField, dstField});
      } else {
         res.fExtraDstFields.emplace_back(&dstField);
      }
   }
   for (const auto &srcField : src.GetTopLevelFields()) {
      const auto dstFieldId = dst.FindFieldId(srcField.GetFieldName());
      if (dstFieldId == ROOT::kInvalidDescriptorId)
         res.fExtraSrcFields.push_back(&srcField);
   }

   // Check compatibility of common fields
   auto fieldsToCheck = commonFields;
   // NOTE: using index-based for loop because the collection may get extended by the iteration
   for (std::size_t fieldIdx = 0; fieldIdx < fieldsToCheck.size(); ++fieldIdx) {
      const auto &field = fieldsToCheck[fieldIdx];

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

      const auto srcRole = field.fSrc->GetStructure();
      const auto dstRole = field.fDst->GetStructure();
      if (srcRole != dstRole) {
         std::stringstream ss;
         ss << "Field `" << field.fSrc->GetFieldName()
            << "` has a different structural role than previously-seen field with the same name (old: " << dstRole
            << ", new: " << srcRole << ")";
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

      // Require that subfields are compatible
      const auto &srcLinks = field.fSrc->GetLinkIds();
      const auto &dstLinks = field.fDst->GetLinkIds();
      if (srcLinks.size() != dstLinks.size()) {
         std::stringstream ss;
         ss << "Field `" << field.fSrc->GetFieldName()
            << "` has a different number of children than previously-seen field with the same name (old: "
            << dstLinks.size() << ", new: " << srcLinks.size() << ")";
         errors.push_back(ss.str());
      } else {
         for (std::size_t linkIdx = 0, linkNum = srcLinks.size(); linkIdx < linkNum; ++linkIdx) {
            const auto &srcSubfield = src.GetFieldDescriptor(srcLinks[linkIdx]);
            const auto &dstSubfield = dst.GetFieldDescriptor(dstLinks[linkIdx]);
            fieldsToCheck.push_back(RCommonField{srcSubfield, dstSubfield});
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

   res.fCommonFields = std::move(commonFields);

   return ROOT::RResult(res);
}

// Applies late model extension to `destination`, adding all `newFields` to it.
[[nodiscard]]
static ROOT::RResult<void>
ExtendDestinationModel(std::span<const ROOT::RFieldDescriptor *> newFields, ROOT::RNTupleModel &dstModel,
                       RNTupleMergeData &mergeData, std::vector<RCommonField> &commonFields)
{
   assert(newFields.size() > 0); // no point in calling this with 0 new cols

   dstModel.Unfreeze();
   ROOT::Internal::RNTupleModelChangeset changeset{dstModel};

   if (mergeData.fMergeOpts.fExtraVerbose) {
      std::string msg = "destination doesn't contain field";
      if (newFields.size() > 1)
         msg += 's';
      msg += ' ';
      msg += std::accumulate(newFields.begin(), newFields.end(), std::string{}, [](const auto &acc, const auto *field) {
         return acc + (acc.length() ? ", " : "") + '`' + field->GetFieldName() + '`';
      });
      R__LOG_INFO(NTupleMergeLog()) << msg << ": adding " << (newFields.size() > 1 ? "them" : "it")
                                    << " to the destination model (entry #" << mergeData.fNumDstEntries << ").";
   }

   changeset.fAddedFields.reserve(newFields.size());
   // First add all non-projected fields...
   for (const auto *fieldDesc : newFields) {
      if (!fieldDesc->IsProjectedField()) {
         auto field = fieldDesc->CreateField(*mergeData.fSrcDescriptor);
         changeset.AddField(std::move(field));
      }
   }
   // ...then add all projected fields.
   for (const auto *fieldDesc : newFields) {
      if (!fieldDesc->IsProjectedField())
         continue;

      ROOT::Internal::RProjectedFields::FieldMap_t fieldMap;
      auto field = fieldDesc->CreateField(*mergeData.fSrcDescriptor);
      const auto sourceId = fieldDesc->GetProjectionSourceId();
      const auto &sourceField = dstModel.GetConstField(mergeData.fSrcDescriptor->GetQualifiedFieldName(sourceId));
      fieldMap[field.get()] = &sourceField;

      for (const auto &subfield : *field) {
         const auto &subFieldDesc = mergeData.fSrcDescriptor->GetFieldDescriptor(subfield.GetOnDiskId());
         const auto subSourceId = subFieldDesc.GetProjectionSourceId();
         const auto &subSourceField =
            dstModel.GetConstField(mergeData.fSrcDescriptor->GetQualifiedFieldName(subSourceId));
         fieldMap[&subfield] = &subSourceField;
      }
      changeset.fAddedProjectedFields.emplace_back(field.get());
      ROOT::Internal::GetProjectedFieldsOfModel(dstModel).Add(std::move(field), fieldMap);
   }
   dstModel.Freeze();
   try {
      mergeData.fDestination.UpdateSchema(changeset, mergeData.fNumDstEntries);
   } catch (const ROOT::RException &ex) {
      return R__FAIL(ex.GetError().GetReport());
   }

   commonFields.reserve(commonFields.size() + newFields.size());
   for (const auto *field : newFields) {
      const auto newFieldInDstId = mergeData.fDstDescriptor.FindFieldId(field->GetFieldName());
      const auto &newFieldInDst = mergeData.fDstDescriptor.GetFieldDescriptor(newFieldInDstId);
      commonFields.emplace_back(*field, newFieldInDst);
   }

   return ROOT::RResult<void>::Success();
}

// Generates default (zero) values for the given columns
[[nodiscard]]
static ROOT::RResult<void>
GenerateZeroPagesForColumns(size_t nEntriesToGenerate, std::span<const RColumnMergeInfo> columns,
                            RSealedPageMergeData &sealedPageData, ROOT::Internal::RPageAllocator &pageAlloc,
                            const ROOT::RNTupleDescriptor &dstDescriptor, const RNTupleMergeData &mergeData)
{
   if (!nEntriesToGenerate)
      return ROOT::RResult<void>::Success();

   for (const auto &column : columns) {
      const ROOT::RFieldDescriptor *field = column.fParentFieldDescriptor;

      // Skip all auxiliary columns
      assert(!field->GetLogicalColumnIds().empty());
      if (field->GetLogicalColumnIds()[0] != column.fInputId)
         continue;

      // Check if this column is a child of a Collection or a Variant. If so, it has no data
      // and can be skipped.
      bool skipColumn = false;
      auto nRepetitions = std::max<std::uint64_t>(field->GetNRepetitions(), 1);
      for (auto parentId = field->GetParentId(); parentId != ROOT::kInvalidDescriptorId;) {
         const ROOT::RFieldDescriptor &parent = column.fParentNTupleDescriptor->GetFieldDescriptor(parentId);
         if (parent.GetStructure() == ROOT::ENTupleStructure::kCollection ||
             parent.GetStructure() == ROOT::ENTupleStructure::kVariant) {
            skipColumn = true;
            break;
         }
         nRepetitions *= std::max<std::uint64_t>(parent.GetNRepetitions(), 1);
         parentId = parent.GetParentId();
      }
      if (skipColumn)
         continue;

      const auto structure = field->GetStructure();

      if (structure == ROOT::ENTupleStructure::kStreamer) {
         return R__FAIL(
            "Destination RNTuple contains a streamer field (" + field->GetFieldName() +
            ") that is not present in one of the sources. "
            "Creating a default value for a streamer field is ill-defined, therefore the merging process will abort.");
      }

      // NOTE: we cannot have a Record here because it has no associated columns.
      R__ASSERT(structure == ROOT::ENTupleStructure::kCollection || structure == ROOT::ENTupleStructure::kVariant ||
                structure == ROOT::ENTupleStructure::kPlain);

      const auto &columnDesc = dstDescriptor.GetColumnDescriptor(column.fOutputId);
      const auto colElement = RColumnElementBase::Generate(columnDesc.GetType());
      const auto nElements = nEntriesToGenerate * nRepetitions;
      const auto nBytesOnStorage = colElement->GetPackedSize(nElements);
      // TODO(gparolini): make this configurable
      constexpr auto kPageSizeLimit = 256 * 1024;
      // TODO(gparolini): consider coalescing the last page if its size is less than some threshold
      const size_t nPages = nBytesOnStorage / kPageSizeLimit + !!(nBytesOnStorage % kPageSizeLimit);
      for (size_t i = 0; i < nPages; ++i) {
         const auto pageSize = (i < nPages - 1) ? kPageSizeLimit : nBytesOnStorage - kPageSizeLimit * (nPages - 1);
         const auto checksumSize = RPageStorage::kNBytesPageChecksum;
         const auto bufSize = pageSize + checksumSize;
         assert(pageSize % colElement->GetSize() == 0);
         const auto nElementsPerPage = pageSize / colElement->GetSize();
         auto page = pageAlloc.NewPage(colElement->GetSize(), nElementsPerPage);
         page.GrowUnchecked(nElementsPerPage);
         memset(page.GetBuffer(), 0, page.GetNBytes());

         auto &buffer = sealedPageData.fBuffers.emplace_back(new unsigned char[bufSize]);
         RPageSink::RSealPageConfig sealConf;
         sealConf.fElement = colElement.get();
         sealConf.fPage = &page;
         sealConf.fBuffer = buffer.get();
         sealConf.fCompressionSettings = mergeData.fMergeOpts.fCompressionSettings.value();
         sealConf.fWriteChecksum = mergeData.fDestination.GetWriteOptions().GetEnablePageChecksums();
         auto sealedPage = RPageSink::SealPage(sealConf);

         sealedPageData.fPagesV.push_back({sealedPage});
         sealedPageData.fGroups.emplace_back(column.fOutputId, sealedPageData.fPagesV.back().cbegin(),
                                             sealedPageData.fPagesV.back().cend());
      }
   }
   return ROOT::RResult<void>::Success();
}

// Merges all columns appearing both in the source and destination RNTuples, just copying them if their
// compression matches ("fast merge") or by unsealing and resealing them with the proper compression.
ROOT::RResult<void>
RNTupleMerger::MergeCommonColumns(ROOT::Internal::RClusterPool &clusterPool,
                                  const ROOT::RClusterDescriptor &clusterDesc,
                                  std::span<const RColumnMergeInfo> commonColumns,
                                  const RCluster::ColumnSet_t &commonColumnSet, std::size_t nCommonColumnsInCluster,
                                  RSealedPageMergeData &sealedPageData, const RNTupleMergeData &mergeData,
                                  ROOT::Internal::RPageAllocator &pageAlloc)
{
   assert(nCommonColumnsInCluster == commonColumnSet.size());
   assert(nCommonColumnsInCluster <= commonColumns.size());
   if (nCommonColumnsInCluster == 0)
      return ROOT::RResult<void>::Success();

   const RCluster *cluster = clusterPool.GetCluster(clusterDesc.GetId(), commonColumnSet);
   // we expect the cluster pool to contain the requested set of columns, since they were
   // validated by CompareDescriptorStructure().
   assert(cluster);

   for (size_t colIdx = 0; colIdx < nCommonColumnsInCluster; ++colIdx) {
      const auto &column = commonColumns[colIdx];
      const auto &columnId = column.fInputId;
      R__ASSERT(clusterDesc.ContainsColumn(columnId));

      const auto &columnDesc = mergeData.fSrcDescriptor->GetColumnDescriptor(columnId);
      const auto srcColElement = column.fInMemoryType
                                    ? ROOT::Internal::GenerateColumnElement(*column.fInMemoryType, columnDesc.GetType())
                                    : RColumnElementBase::Generate(columnDesc.GetType());
      const auto dstColElement = column.fInMemoryType
                                    ? ROOT::Internal::GenerateColumnElement(*column.fInMemoryType, column.fColumnType)
                                    : RColumnElementBase::Generate(column.fColumnType);

      // Now get the pages for this column in this cluster
      const auto &pages = clusterDesc.GetPageRange(columnId);

      RPageStorage::SealedPageSequence_t sealedPages;
      sealedPages.resize(pages.GetPageInfos().size());

      // Each column range potentially has a distinct compression settings
      const auto colRangeCompressionSettings = clusterDesc.GetColumnRange(columnId).GetCompressionSettings().value();
      // If either the compression or the encoding of the source doesn't match that of the destination, we need
      // to reseal the page. Otherwise, if both match, we can fast merge.
      const bool needsResealing =
         colRangeCompressionSettings != mergeData.fMergeOpts.fCompressionSettings.value() ||
         srcColElement->GetIdentifier().fOnDiskType != dstColElement->GetIdentifier().fOnDiskType;

      if (needsResealing && mergeData.fMergeOpts.fExtraVerbose) {
         R__LOG_INFO(NTupleMergeLog())
            << "Resealing column " << column.fColumnName << ": { compression: " << colRangeCompressionSettings << " => "
            << mergeData.fMergeOpts.fCompressionSettings.value()
            << ", onDiskType: " << RColumnElementBase::GetColumnTypeName(srcColElement->GetIdentifier().fOnDiskType)
            << " => " << RColumnElementBase::GetColumnTypeName(dstColElement->GetIdentifier().fOnDiskType);
      }

      size_t pageBufferBaseIdx = sealedPageData.fBuffers.size();
      // If the column range already has the right compression we don't need to allocate any new buffer, so we don't
      // bother reserving memory for them.
      if (needsResealing)
         sealedPageData.fBuffers.resize(sealedPageData.fBuffers.size() + pages.GetPageInfos().size());

      // If this column is deferred, we may need to fill "holes" until its real start. We fill any missing entry
      // with zeroes, like we do for extraDstColumns.
      // As an optimization, we don't do this for the first source (since we can rely on the FirstElementIndex and
      // deferred column mechanism in that case).
      // TODO: also avoid doing this if we added no real page of this column to the destination yet.
      if (columnDesc.GetFirstElementIndex() > clusterDesc.GetFirstEntryIndex() && mergeData.fNumDstEntries > 0) {
         const auto nMissingEntries = columnDesc.GetFirstElementIndex() - clusterDesc.GetFirstEntryIndex();
         auto res = GenerateZeroPagesForColumns(nMissingEntries, {&column, 1}, sealedPageData, pageAlloc,
                                                mergeData.fDstDescriptor, mergeData);
         if (!res)
            return R__FORWARD_ERROR(res);
      }

      // Loop over the pages
      std::uint64_t pageIdx = 0;
      for (const auto &pageInfo : pages.GetPageInfos()) {
         assert(pageIdx < sealedPages.size());
         assert(sealedPageData.fBuffers.size() == 0 || pageIdx < sealedPageData.fBuffers.size());
         assert(pageInfo.GetLocator().GetType() != RNTupleLocator::kTypePageZero);

         ROOT::Internal::ROnDiskPage::Key key{columnId, pageIdx};
         auto onDiskPage = cluster->GetOnDiskPage(key);

         const auto checksumSize = pageInfo.HasChecksum() * RPageStorage::kNBytesPageChecksum;
         RPageStorage::RSealedPage &sealedPage = sealedPages[pageIdx];
         sealedPage.SetNElements(pageInfo.GetNElements());
         sealedPage.SetHasChecksum(pageInfo.HasChecksum());
         sealedPage.SetBufferSize(pageInfo.GetLocator().GetNBytesOnStorage() + checksumSize);
         sealedPage.SetBuffer(onDiskPage->GetAddress());
         // TODO(gparolini): more graceful error handling (skip the page?)
         sealedPage.VerifyChecksumIfEnabled().ThrowOnError();
         R__ASSERT(onDiskPage && (onDiskPage->GetSize() == sealedPage.GetBufferSize()));

         if (needsResealing) {
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

   return ROOT::RResult<void>::Success();
}

// Iterates over all clusters of `source` and merges their pages into `destination`.
// It is assumed that all columns in `commonColumns` are present (and compatible) in both the source and
// the destination's schemas.
// The pages may be "fast-merged" (i.e. simply copied with no decompression/recompression) if the target
// compression is unspecified or matches the original compression settings.
ROOT::RResult<void>
RNTupleMerger::MergeSourceClusters(RPageSource &source, std::span<const RColumnMergeInfo> commonColumns,
                                   std::span<const RColumnMergeInfo> extraDstColumns, RNTupleMergeData &mergeData)
{
   ROOT::Internal::RClusterPool clusterPool{source};

   std::vector<RColumnMergeInfo> missingColumns{extraDstColumns.begin(), extraDstColumns.end()};

   // Loop over all clusters in this file.
   // descriptor->GetClusterIterable() doesn't guarantee any specific order, so we explicitly
   // request the first cluster.
   ROOT::DescriptorId_t clusterId = mergeData.fSrcDescriptor->FindClusterId(0, 0);
   while (clusterId != ROOT::kInvalidDescriptorId) {
      const auto &clusterDesc = mergeData.fSrcDescriptor->GetClusterDescriptor(clusterId);
      const auto nClusterEntries = clusterDesc.GetNEntries();
      R__ASSERT(nClusterEntries > 0);

      // NOTE: just because a column is in `commonColumns` it doesn't mean that each cluster in the source contains it,
      // as it may be a deferred column that only has real data in a future cluster.
      // We need to figure out which columns are actually present in this cluster so we only merge their pages (the
      // missing columns are handled by synthesizing zero pages - see below).
      size_t nCommonColumnsInCluster = commonColumns.size();
      while (nCommonColumnsInCluster > 0) {
         // Since `commonColumns` is sorted by column input id, we can simply traverse it from the back and stop as
         // soon as we find a common column that appears in this cluster: we know that in that case all previous
         // columns must appear as well.
         if (clusterDesc.ContainsColumn(commonColumns[nCommonColumnsInCluster - 1].fInputId))
            break;
         --nCommonColumnsInCluster;
      }

      // Convert columns to a ColumnSet for the ClusterPool query
      RCluster::ColumnSet_t commonColumnSet;
      commonColumnSet.reserve(nCommonColumnsInCluster);
      for (size_t i = 0; i < nCommonColumnsInCluster; ++i)
         commonColumnSet.emplace(commonColumns[i].fInputId);

      // For each cluster, the "missing columns" are the union of the extraDstColumns and the common columns
      // that are not present in the cluster. We generate zero pages for all of them.
      missingColumns.resize(extraDstColumns.size());
      for (size_t i = nCommonColumnsInCluster; i < commonColumns.size(); ++i)
         missingColumns.push_back(commonColumns[i]);

      RSealedPageMergeData sealedPageData;
      auto res = MergeCommonColumns(clusterPool, clusterDesc, commonColumns, commonColumnSet, nCommonColumnsInCluster,
                                    sealedPageData, mergeData, *fPageAlloc);
      if (!res)
         return R__FORWARD_ERROR(res);

      res = GenerateZeroPagesForColumns(nClusterEntries, missingColumns, sealedPageData, *fPageAlloc,
                                        mergeData.fDstDescriptor, mergeData);
      if (!res)
         return R__FORWARD_ERROR(res);

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
   return ROOT::RResult<void>::Success();
}

static std::optional<std::type_index> ColumnInMemoryType(std::string_view fieldType, ENTupleColumnType onDiskType)
{
   if (onDiskType == ENTupleColumnType::kIndex32 || onDiskType == ENTupleColumnType::kSplitIndex32 ||
       onDiskType == ENTupleColumnType::kIndex64 || onDiskType == ENTupleColumnType::kSplitIndex64)
      return typeid(ROOT::Internal::RColumnIndex);

   if (onDiskType == ENTupleColumnType::kSwitch)
      return typeid(ROOT::Internal::RColumnSwitch);

   // clang-format off
   if (fieldType == "bool")          return typeid(bool);
   if (fieldType == "std::byte")     return typeid(std::byte);
   if (fieldType == "char")          return typeid(char);
   if (fieldType == "std::int8_t")   return typeid(std::int8_t);
   if (fieldType == "std::uint8_t")  return typeid(std::uint8_t);
   if (fieldType == "std::int16_t")  return typeid(std::int16_t);
   if (fieldType == "std::uint16_t") return typeid(std::uint16_t);
   if (fieldType == "std::int32_t")  return typeid(std::int32_t);
   if (fieldType == "std::uint32_t") return typeid(std::uint32_t);
   if (fieldType == "std::int64_t")  return typeid(std::int64_t);
   if (fieldType == "std::uint64_t") return typeid(std::uint64_t);
   if (fieldType == "float")         return typeid(float);
   if (fieldType == "double")        return typeid(double);
   // clang-format on

   // if the type is not one of those above, we use the default in-memory type.
   return std::nullopt;
}

// Given a field, fill `columns` and `mergeData.fColumnIdMap` with information about all columns belonging to it and its
// subfields. `mergeData.fColumnIdMap` is used to map matching columns from different sources to the same output column
// in the destination. We match columns by their "fully qualified name", which is the concatenation of their ancestor
// fields' names and the column index. By this point, since we called `CompareDescriptorStructure()` earlier, we should
// be guaranteed that two matching columns will have at least compatible representations. NOTE: srcFieldDesc and
// dstFieldDesc may alias.
static void AddColumnsFromField(std::vector<RColumnMergeInfo> &columns, const ROOT::RNTupleDescriptor &srcDesc,
                                RNTupleMergeData &mergeData, const ROOT::RFieldDescriptor &srcFieldDesc,
                                const ROOT::RFieldDescriptor &dstFieldDesc, const std::string &prefix = "")
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
      // NOTE(gparolini): the parent field is used when synthesizing zero pages, which happens in 2 situations:
      // 1. when adding extra dst columns (in which case we need to synthesize zero pages for the incoming src), and
      // 2. when merging a deferred column into an existing column (in which case we need to fill the "hole" with
      // zeroes). For the first case srcFieldDesc and dstFieldDesc are the same (see the calling site of this function),
      // but for the second case they're not, and we need to pick the source field because we will then check the
      // column's *input* id inside fParentFieldDescriptor to see if it's a suppressed column (see
      // GenerateZeroPagesForColumns()).
      info.fParentFieldDescriptor = &srcFieldDesc;
      // Save the parent field descriptor since this may be either the source or destination descriptor depending on
      // whether this is an extraDstField or a commonField. We will need this in GenerateZeroPagesForColumns() to
      // properly walk up the field hierarchy.
      info.fParentNTupleDescriptor = &srcDesc;

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
         R__LOG_INFO(NTupleMergeLog()) << "Adding column " << info.fColumnName << " with log.id " << srcColumnId
                                       << ", phys.id " << srcColumn.GetPhysicalId() << ", type "
                                       << RColumnElementBase::GetColumnTypeName(srcColumn.GetType()) << " -> log.id "
                                       << info.fOutputId << ", type "
                                       << RColumnElementBase::GetColumnTypeName(info.fColumnType);
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
// While doing so, it collects such information in `mergeData.fColumnIdMap`, which is used by later calls to this
// function to map already-seen column names to their chosen outputId, type and so on.
static RColumnInfoGroup GatherColumnInfos(const RDescriptorsComparison &descCmp, const ROOT::RNTupleDescriptor &srcDesc,
                                          RNTupleMergeData &mergeData)
{
   RColumnInfoGroup res;
   for (const ROOT::RFieldDescriptor *field : descCmp.fExtraDstFields) {
      AddColumnsFromField(res.fExtraDstColumns, mergeData.fDstDescriptor, mergeData, *field, *field);
   }
   for (const auto &[srcField, dstField] : descCmp.fCommonFields) {
      AddColumnsFromField(res.fCommonColumns, srcDesc, mergeData, *srcField, *dstField);
   }

   // Sort the commonColumns by ID so we can more easily tell how many common columns each cluster has
   // (since each cluster must contain all columns of the previous cluster plus potentially some new ones)
   std::sort(res.fCommonColumns.begin(), res.fCommonColumns.end(),
             [](const auto &a, const auto &b) { return a.fInputId < b.fInputId; });

   return res;
}

static void PrefillColumnMap(const ROOT::RNTupleDescriptor &desc, const ROOT::RFieldDescriptor &fieldDesc,
                             ColumnIdMap_t &colIdMap, const std::string &prefix = "")
{
   std::string name = prefix + '.' + fieldDesc.GetFieldName();
   for (const auto &colId : fieldDesc.GetLogicalColumnIds()) {
      const auto &colDesc = desc.GetColumnDescriptor(colId);
      RColumnOutInfo info{};
      const auto colName = name + '.' + std::to_string(colDesc.GetIndex());
      info.fColumnId = colDesc.GetLogicalId();
      info.fColumnType = colDesc.GetType();
      colIdMap[colName] = info;
   }

   for (const auto &subId : fieldDesc.GetLinkIds()) {
      const auto &subfield = desc.GetFieldDescriptor(subId);
      PrefillColumnMap(desc, subfield, colIdMap, name);
   }
}

RNTupleMerger::RNTupleMerger(std::unique_ptr<ROOT::Internal::RPagePersistentSink> destination,
                             std::unique_ptr<ROOT::RNTupleModel> model)
   // TODO(gparolini): consider using an arena allocator instead, since we know the precise lifetime
   // of the RNTuples we are going to handle (e.g. we can reset the arena at every source)
   : fDestination(std::move(destination)),
     fPageAlloc(std::make_unique<ROOT::Internal::RPageAllocatorHeap>()),
     fModel(std::move(model))
{
   R__ASSERT(fDestination);

#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled())
      fTaskGroup = TTaskGroup();
#endif
}

RNTupleMerger::RNTupleMerger(std::unique_ptr<ROOT::Internal::RPagePersistentSink> destination)
   : RNTupleMerger(std::move(destination), nullptr)
{
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

   // we should have a model if and only if the destination is initialized.
   if (!!fModel != fDestination->IsInitialized()) {
      return R__FAIL(
         "passing an already-initialized destination to RNTupleMerger::Merge (i.e. trying to do incremental "
         "merging) can only be done by providing a valid ROOT::RNTupleModel when constructing the RNTupleMerger.");
   }

   RNTupleMergeData mergeData{sources, *fDestination, mergeOpts};
   mergeData.fNumDstEntries = mergeData.fDestination.GetNEntries();

   if (fModel) {
      // If this is an incremental merging, pre-fill the column id map with the existing destination ids.
      // Otherwise we would generate new output ids that may not match the ones in the destination!
      for (const auto &field : mergeData.fDstDescriptor.GetTopLevelFields()) {
         PrefillColumnMap(fDestination->GetDescriptor(), field, mergeData.fColumnIdMap);
      }
   }

#define SKIP_OR_ABORT(errMsg)                                                         \
   do {                                                                               \
      if (mergeOpts.fErrBehavior == ENTupleMergeErrBehavior::kSkip) {                 \
         R__LOG_WARNING(NTupleMergeLog()) << "Skipping RNTuple due to: " << (errMsg); \
         continue;                                                                    \
      } else {                                                                        \
         return R__FAIL(errMsg);                                                      \
      }                                                                               \
   } while (0)

   // Merge main loop
   for (RPageSource *source : sources) {
      // We need to make sure the streamer info from the source files is loaded otherwise we may not be able
      // to build the streamer info of user-defined types unless we have their dictionaries available.
      source->LoadStreamerInfo();

      source->Attach(RNTupleSerializer::EDescriptorDeserializeMode::kForWriting);
      auto srcDescriptor = source->GetSharedDescriptorGuard();
      mergeData.fSrcDescriptor = &srcDescriptor.GetRef();

      // Create sink from the input model if not initialized
      if (!fModel) {
         fModel = fDestination->InitFromDescriptor(srcDescriptor.GetRef(), false /* copyClusters */);
      }

      for (const auto &extraTypeInfoDesc : srcDescriptor->GetExtraTypeInfoIterable())
         fDestination->UpdateExtraTypeInfo(extraTypeInfoDesc);

      auto descCmpRes = CompareDescriptorStructure(mergeData.fDstDescriptor, srcDescriptor.GetRef());
      if (!descCmpRes) {
         SKIP_OR_ABORT(std::string("Source RNTuple has an incompatible schema with the destination:\n") +
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
            auto res = ExtendDestinationModel(descCmp.fExtraSrcFields, *fModel, mergeData, descCmp.fCommonFields);
            if (!res)
               return R__FORWARD_ERROR(res);
         } else if (mergeOpts.fMergingMode == ENTupleMergingMode::kStrict) {
            // If the current source has extra fields and we're in Strict mode, error
            std::string msg = "Source RNTuple has extra fields that the destination RNTuple doesn't have:";
            for (const auto *field : descCmp.fExtraSrcFields) {
               msg += "\n  " + field->GetFieldName() + " : " + field->GetTypeName();
            }
            SKIP_OR_ABORT(msg);
         }
      }

      // handle extra dst fields & common fields
      auto columnInfos = GatherColumnInfos(descCmp, srcDescriptor.GetRef(), mergeData);
      auto res = MergeSourceClusters(*source, columnInfos.fCommonColumns, columnInfos.fExtraDstColumns, mergeData);
      if (!res)
         return R__FORWARD_ERROR(res);
   } // end loop over sources

   if (fDestination->GetNEntries() == 0)
      R__LOG_WARNING(NTupleMergeLog()) << "Output RNTuple '" << fDestination->GetNTupleName() << "' has no entries.";

   // Commit the output
   fDestination->CommitClusterGroup();
   fDestination->CommitDataset();

   return RResult<void>::Success();
}
