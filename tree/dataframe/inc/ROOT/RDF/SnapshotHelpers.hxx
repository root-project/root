/**
 \file ROOT/RDF/SnapshotHelpers.hxx
 \ingroup dataframe
 \author Enrico Guiraud, CERN
 \author Danilo Piparo, CERN
 \date 2016-12
 \author Vincenzo Eduardo Padulano
 \author Stephan Hageboeck
 \date 2025-06
*/

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RDF_SNAPSHOTHELPERS
#define RDF_SNAPSHOTHELPERS

#include <ROOT/RSnapshotOptions.hxx>

#include <ROOT/RDF/RActionImpl.hxx>
#include <ROOT/RDF/RLoopManager.hxx>
#include <ROOT/RDF/Utils.hxx>

#include <array>
#include <memory>
#include <variant>

class TBranch;
class TFile;

namespace ROOT {
class REntry;
class RFieldToken;
class RNTupleFillContext;
class RNTupleParallelWriter;
class TBufferMerger;
class TBufferMergerFile;
} // namespace ROOT

namespace ROOT::Internal::RDF {

class R__CLING_PTRCHECK(off) UntypedSnapshotRNTupleHelper final : public RActionImpl<UntypedSnapshotRNTupleHelper> {
   std::string fFileName;
   std::string fDirName;
   std::string fNTupleName;

   std::unique_ptr<TFile> fOutputFile;

   RSnapshotOptions fOptions;
   ROOT::Detail::RDF::RLoopManager *fInputLoopManager;
   ROOT::Detail::RDF::RLoopManager *fOutputLoopManager;
   ColumnNames_t fInputFieldNames; // This contains the resolved aliases
   ColumnNames_t fOutputFieldNames;
   std::unique_ptr<ROOT::RNTupleParallelWriter> fWriter;
   std::vector<ROOT::RFieldToken> fFieldTokens;

   unsigned int fNSlots;
   std::vector<std::shared_ptr<ROOT::RNTupleFillContext>> fFillContexts;
   std::vector<std::unique_ptr<ROOT::REntry>> fEntries;

   std::vector<const std::type_info *> fInputColumnTypeIDs; // Types for the input columns

public:
   UntypedSnapshotRNTupleHelper(unsigned int nSlots, std::string_view filename, std::string_view dirname,
                                std::string_view ntuplename, const ColumnNames_t &vfnames, const ColumnNames_t &fnames,
                                const RSnapshotOptions &options, ROOT::Detail::RDF::RLoopManager *inputLM,
                                ROOT::Detail::RDF::RLoopManager *outputLM,
                                const std::vector<const std::type_info *> &colTypeIDs);

   UntypedSnapshotRNTupleHelper(const UntypedSnapshotRNTupleHelper &) = delete;
   UntypedSnapshotRNTupleHelper &operator=(const UntypedSnapshotRNTupleHelper &) = delete;
   UntypedSnapshotRNTupleHelper(UntypedSnapshotRNTupleHelper &&) noexcept;
   UntypedSnapshotRNTupleHelper &operator=(UntypedSnapshotRNTupleHelper &&) noexcept;
   ~UntypedSnapshotRNTupleHelper() final;

   void Initialize();

   void Exec(unsigned int slot, const std::vector<void *> &values);

   void InitTask(TTreeReader *, unsigned int slot);

   void FinalizeTask(unsigned int slot);

   void Finalize();

   std::string GetActionName() { return "Snapshot"; }

   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return [](unsigned int, const RSampleInfo &) mutable {};
   }

   UntypedSnapshotRNTupleHelper MakeNew(void *newName);
};

/// Stores properties of each output branch in a Snapshot.
struct RBranchData {
   /// Stores variations of a fundamental type.
   /// The bytes hold anything up to double or 64-bit numbers, and are cleared for every event.
   /// This allows for binding the branches directly to these bytes.
   struct FundamentalType {
      static constexpr std::size_t fNBytes = 8;
      alignas(8) std::array<std::byte, fNBytes> fBytes{std::byte{0}}; // 8 bytes to store any fundamental type
      unsigned short fSize = 0;
      FundamentalType(unsigned short size) : fSize(size) { assert(size <= fNBytes); }
   };
   /// Stores empty instances of classes, so a dummy object can be written when a systematic variation
   /// doesn't pass a selection cut.
   struct EmptyDynamicType {
      const TClass *fTClass = nullptr;
      std::shared_ptr<void> fEmptyInstance = nullptr;
      void *fRawPtrToEmptyInstance = nullptr; // Needed because TTree expects pointer to pointer
   };

   std::string fInputBranchName; // This contains resolved aliases
   std::string fOutputBranchName;
   const std::type_info *fInputTypeID = nullptr;
   TBranch *fOutputBranch = nullptr;
   void *fBranchAddressForCArrays = nullptr; // Used to detect if branch addresses need to be updated

   int fVariationIndex = -1; // For branches that are only valid if a specific filter passed
   std::variant<FundamentalType, EmptyDynamicType> fTypeData = FundamentalType{0};
   bool fIsCArray = false;
   bool fIsDefine = false;

   RBranchData() = default;
   RBranchData(std::string inputBranchName, std::string outputBranchName, bool isDefine, const std::type_info *typeID);

   void ClearBranchPointers()
   {
      fOutputBranch = nullptr;
      fBranchAddressForCArrays = nullptr;
   }
   void *EmptyInstance(bool pointerToPointer);
   void ClearBranchContents();
   /// For fundamental types represented by TDataType, fetch a value from the pointer into the local branch buffer.
   /// If the branch holds a class type, nothing happens.
   /// \return true if the branch holds a fundamental type, false if it holds a class type.
   bool WriteValueIfFundamental(void *valuePtr)
   {
      if (auto fundamentalType = std::get_if<FundamentalType>(&fTypeData); fundamentalType) {
         std::memcpy(fundamentalType->fBytes.data(), valuePtr, fundamentalType->fSize);
         return true;
      }
      return false;
   }
};

class R__CLING_PTRCHECK(off) UntypedSnapshotTTreeHelper final : public RActionImpl<UntypedSnapshotTTreeHelper> {
   std::string fFileName;
   std::string fDirName;
   std::string fTreeName;
   RSnapshotOptions fOptions;
   std::unique_ptr<TFile> fOutputFile;
   std::unique_ptr<TTree> fOutputTree; // must be a ptr because TTrees are not copy/move constructible
   bool fBranchAddressesNeedReset{true};
   TTree *fInputTree = nullptr; // Current input tree. Set at initialization time (`InitTask`)
   std::vector<RBranchData> fBranchData; // Information for all output branches
   ROOT::Detail::RDF::RLoopManager *fOutputLoopManager;
   ROOT::Detail::RDF::RLoopManager *fInputLoopManager;

public:
   UntypedSnapshotTTreeHelper(std::string_view filename, std::string_view dirname, std::string_view treename,
                              const ColumnNames_t &vbnames, const ColumnNames_t &bnames,
                              const RSnapshotOptions &options, std::vector<bool> &&isDefine,
                              ROOT::Detail::RDF::RLoopManager *loopManager, ROOT::Detail::RDF::RLoopManager *inputLM,
                              const std::vector<const std::type_info *> &colTypeIDs);

   UntypedSnapshotTTreeHelper(const UntypedSnapshotTTreeHelper &) = delete;
   UntypedSnapshotTTreeHelper &operator=(const UntypedSnapshotTTreeHelper &) = delete;
   UntypedSnapshotTTreeHelper(UntypedSnapshotTTreeHelper &&) noexcept;
   UntypedSnapshotTTreeHelper &operator=(UntypedSnapshotTTreeHelper &&) noexcept;
   ~UntypedSnapshotTTreeHelper() final;

   void InitTask(TTreeReader *, unsigned int);

   void Exec(unsigned int, const std::vector<void *> &values);

   void UpdateCArraysPtrs(const std::vector<void *> &values);

   void SetBranches(const std::vector<void *> &values);

   void SetEmptyBranches(TTree *inputTree, TTree &outputTree);

   void Initialize();

   void Finalize();

   std::string GetActionName() { return "Snapshot"; }

   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return [this](unsigned int, const RSampleInfo &) mutable { fBranchAddressesNeedReset = true; };
   }

   UntypedSnapshotTTreeHelper MakeNew(void *newName, std::string_view /*variation*/ = "nominal");
};

class R__CLING_PTRCHECK(off) UntypedSnapshotTTreeHelperMT final : public RActionImpl<UntypedSnapshotTTreeHelperMT> {

   // IMT-specific data members

   unsigned int fNSlots;
   std::unique_ptr<ROOT::TBufferMerger> fMerger; // must use a ptr because TBufferMerger is not movable
   std::vector<std::shared_ptr<ROOT::TBufferMergerFile>> fOutputFiles;
   std::vector<std::unique_ptr<TTree>> fOutputTrees;
   std::vector<int> fBranchAddressesNeedReset; // vector<bool> does not allow concurrent writing of different elements
   std::vector<TTree *> fInputTrees; // Current input trees, one per slot. Set at initialization time (`InitTask`)
   std::vector<std::vector<RBranchData>> fBranchData; // Information for all output branches of each slot

   // Attributes of the output TTree

   std::string fFileName;
   std::string fDirName;
   std::string fTreeName;
   TFile *fOutputFile; // Non-owning view on the output file
   RSnapshotOptions fOptions;

   // Attributes related to the computation graph

   ROOT::Detail::RDF::RLoopManager *fOutputLoopManager;
   ROOT::Detail::RDF::RLoopManager *fInputLoopManager;

public:
   UntypedSnapshotTTreeHelperMT(unsigned int nSlots, std::string_view filename, std::string_view dirname,
                                std::string_view treename, const ColumnNames_t &vbnames, const ColumnNames_t &bnames,
                                const RSnapshotOptions &options, std::vector<bool> &&isDefine,
                                ROOT::Detail::RDF::RLoopManager *loopManager, ROOT::Detail::RDF::RLoopManager *inputLM,
                                const std::vector<const std::type_info *> &colTypeIDs);

   UntypedSnapshotTTreeHelperMT(const UntypedSnapshotTTreeHelperMT &) = delete;
   UntypedSnapshotTTreeHelperMT &operator=(const UntypedSnapshotTTreeHelperMT &) = delete;
   UntypedSnapshotTTreeHelperMT(UntypedSnapshotTTreeHelperMT &&) noexcept;
   UntypedSnapshotTTreeHelperMT &operator=(UntypedSnapshotTTreeHelperMT &&) noexcept;
   ~UntypedSnapshotTTreeHelperMT() final;

   void InitTask(TTreeReader *r, unsigned int slot);

   void FinalizeTask(unsigned int slot);

   void Exec(unsigned int slot, const std::vector<void *> &values);

   void UpdateCArraysPtrs(unsigned int slot, const std::vector<void *> &values);

   void SetBranches(unsigned int slot, const std::vector<void *> &values);

   void SetEmptyBranches(TTree *inputTree, TTree &outputTree);

   void Initialize();

   void Finalize();

   std::string GetActionName() { return "Snapshot"; }

   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return [this](unsigned int slot, const RSampleInfo &) mutable { fBranchAddressesNeedReset[slot] = 1; };
   }

   UntypedSnapshotTTreeHelperMT MakeNew(void *newName, std::string_view /*variation*/ = "nominal");
};

struct SnapshotOutputWriter;

/// TTree snapshot helper with systematic variations.
class R__CLING_PTRCHECK(off) SnapshotHelperWithVariations
   : public ROOT::Detail::RDF::RActionImpl<SnapshotHelperWithVariations> {
   RSnapshotOptions fOptions;
   std::shared_ptr<SnapshotOutputWriter> fOutputHandle;
   TTree *fInputTree = nullptr; // Current input tree. Set at initialization time (`InitTask`)
   std::vector<RBranchData> fBranchData;
   ROOT::Detail::RDF::RLoopManager *fInputLoopManager = nullptr;
   ROOT::Detail::RDF::RLoopManager *fOutputLoopManager = nullptr;

   void ClearOutputBranches();

public:
   SnapshotHelperWithVariations(std::string_view filename, std::string_view dirname, std::string_view treename,
                                const ColumnNames_t & /*vbnames*/, const ColumnNames_t &bnames,
                                const RSnapshotOptions &options, std::vector<bool> && /*isDefine*/,
                                ROOT::Detail::RDF::RLoopManager *outputLoopMgr,
                                ROOT::Detail::RDF::RLoopManager *inputLoopMgr,
                                const std::vector<const std::type_info *> &colTypeIDs);

   SnapshotHelperWithVariations(SnapshotHelperWithVariations const &) noexcept = delete;
   SnapshotHelperWithVariations(SnapshotHelperWithVariations &&) noexcept = default;
   ~SnapshotHelperWithVariations() = default;
   SnapshotHelperWithVariations &operator=(SnapshotHelperWithVariations const &) = delete;
   SnapshotHelperWithVariations &operator=(SnapshotHelperWithVariations &&) noexcept = default;

   void RegisterVariedColumn(unsigned int slot, unsigned int columnIndex, unsigned int originalColumnIndex,
                             unsigned int varationIndex, std::string const &variationName);

   void InitTask(TTreeReader *, unsigned int slot);

   void Exec(unsigned int /*slot*/, const std::vector<void *> &values, std::vector<bool> const &filterPassed);

   /// Nothing to do. All initialisations run in the constructor or InitTask().
   void Initialize() {}

   void Finalize();

   std::string GetActionName() { return "SnapshotWithVariations"; }
};

} // namespace ROOT::Internal::RDF

#endif
