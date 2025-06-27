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

class TBranch;
class TFile;

namespace ROOT {
class RNTupleWriter;
class REntry;
class TBufferMerger;
class TBufferMergerFile;
} // namespace ROOT

namespace ROOT::Internal::RDF {

class RBranchSet {
   std::vector<TBranch *> fBranches;
   std::vector<std::string> fNames;
   std::vector<bool> fIsCArray;

public:
   TBranch *Get(const std::string &name) const;
   bool IsCArray(const std::string &name) const;
   void Insert(const std::string &name, TBranch *address, bool isCArray = false);
   void Clear();
   void AssertNoNullBranchAddresses();
};

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
   std::unique_ptr<ROOT::RNTupleWriter> fWriter;

   ROOT::REntry *fOutputEntry;

   std::vector<bool> fIsDefine;

   std::vector<const std::type_info *> fInputColumnTypeIDs; // Types for the input columns

public:
   UntypedSnapshotRNTupleHelper(std::string_view filename, std::string_view dirname, std::string_view ntuplename,
                                const ColumnNames_t &vfnames, const ColumnNames_t &fnames,
                                const RSnapshotOptions &options, ROOT::Detail::RDF::RLoopManager *inputLM,
                                ROOT::Detail::RDF::RLoopManager *outputLM, std::vector<bool> &&isDefine,
                                const std::vector<const std::type_info *> &colTypeIDs);

   UntypedSnapshotRNTupleHelper(const UntypedSnapshotRNTupleHelper &) = delete;
   UntypedSnapshotRNTupleHelper &operator=(const UntypedSnapshotRNTupleHelper &) = delete;
   UntypedSnapshotRNTupleHelper(UntypedSnapshotRNTupleHelper &&) noexcept;
   UntypedSnapshotRNTupleHelper &operator=(UntypedSnapshotRNTupleHelper &&) noexcept;
   ~UntypedSnapshotRNTupleHelper() final;

   void InitTask(TTreeReader *, unsigned int /* slot */) {}

   void Exec(unsigned int /* slot */, const std::vector<void *> &values);

   void Initialize();

   void Finalize();

   std::string GetActionName() { return "Snapshot"; }

   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return [](unsigned int, const RSampleInfo &) mutable {};
   }

   UntypedSnapshotRNTupleHelper MakeNew(void *newName);
};

class R__CLING_PTRCHECK(off) UntypedSnapshotTTreeHelper final : public RActionImpl<UntypedSnapshotTTreeHelper> {
   std::string fFileName;
   std::string fDirName;
   std::string fTreeName;
   RSnapshotOptions fOptions;
   std::unique_ptr<TFile> fOutputFile;
   std::unique_ptr<TTree> fOutputTree; // must be a ptr because TTrees are not copy/move constructible
   bool fBranchAddressesNeedReset{true};
   ColumnNames_t fInputBranchNames; // This contains the resolved aliases
   ColumnNames_t fOutputBranchNames;
   TTree *fInputTree = nullptr; // Current input tree. Set at initialization time (`InitTask`)
   // TODO we might be able to unify fBranches, fBranchAddresses and fOutputBranches
   std::vector<TBranch *> fBranches;     // Addresses of branches in output, non-null only for the ones holding C arrays
   std::vector<void *> fBranchAddresses; // Addresses of objects associated to output branches
   RBranchSet fOutputBranches;
   std::vector<bool> fIsDefine;
   ROOT::Detail::RDF::RLoopManager *fOutputLoopManager;
   ROOT::Detail::RDF::RLoopManager *fInputLoopManager;
   std::vector<const std::type_info *> fInputColumnTypeIDs; // Types for the input columns

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
   // Addresses of branches in output per slot, non-null only for the ones holding C arrays
   std::vector<std::vector<TBranch *>> fBranches;
   // Addresses of objects associated to output branches per slot, non-null only for the ones holding C arrays
   std::vector<std::vector<void *>> fBranchAddresses;
   std::vector<RBranchSet> fOutputBranches; // Unique set of output branches, one per slot.

   // Attributes of the output TTree

   std::string fFileName;
   std::string fDirName;
   std::string fTreeName;
   TFile *fOutputFile; // Non-owning view on the output file
   RSnapshotOptions fOptions;
   std::vector<std::string> fOutputBranchNames;

   // Attributes related to the computation graph

   ROOT::Detail::RDF::RLoopManager *fOutputLoopManager;
   ROOT::Detail::RDF::RLoopManager *fInputLoopManager;
   std::vector<std::string> fInputBranchNames;              // This contains the resolved aliases
   std::vector<const std::type_info *> fInputColumnTypeIDs; // Types for the input columns

   std::vector<bool> fIsDefine;

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

} // namespace ROOT::Internal::RDF

#endif
