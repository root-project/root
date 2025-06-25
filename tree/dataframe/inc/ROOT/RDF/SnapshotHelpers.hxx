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

#include "ROOT/RSnapshotOptions.hxx"

#include "ROOT/RDF/RActionImpl.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/Utils.hxx"

#include "ROOT/RNTupleDS.hxx"
#include "ROOT/RTTreeDS.hxx"
#include "ROOT/TBufferMerger.hxx"
#include "ROOT/RNTupleWriter.hxx"

#include <TClassEdit.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TTree.h>
#include <TTreeReader.h>

namespace ROOT::Internal::RDF {
template <typename T>
void *GetData(ROOT::VecOps::RVec<T> &v)
{
   return v.data();
}

template <typename T>
void *GetData(T & /*v*/)
{
   return nullptr;
}

class RBranchSet {
   std::vector<TBranch *> fBranches;
   std::vector<std::string> fNames;
   std::vector<bool> fIsCArray;

public:
   TBranch *Get(const std::string &name) const
   {
      auto it = std::find(fNames.begin(), fNames.end(), name);
      if (it == fNames.end())
         return nullptr;
      return fBranches[std::distance(fNames.begin(), it)];
   }

   bool IsCArray(const std::string &name) const
   {
      if (auto it = std::find(fNames.begin(), fNames.end(), name); it != fNames.end())
         return fIsCArray[std::distance(fNames.begin(), it)];
      return false;
   }

   void Insert(const std::string &name, TBranch *address, bool isCArray = false)
   {
      if (address == nullptr) {
         throw std::logic_error("Trying to insert a null branch address.");
      }
      if (std::find(fBranches.begin(), fBranches.end(), address) != fBranches.end()) {
         throw std::logic_error("Trying to insert a branch address that's already present.");
      }
      if (std::find(fNames.begin(), fNames.end(), name) != fNames.end()) {
         throw std::logic_error("Trying to insert a branch name that's already present.");
      }
      fNames.emplace_back(name);
      fBranches.emplace_back(address);
      fIsCArray.push_back(isCArray);
   }

   void Clear()
   {
      fBranches.clear();
      fNames.clear();
      fIsCArray.clear();
   }

   void AssertNoNullBranchAddresses()
   {
      std::vector<TBranch *> branchesWithNullAddress;
      std::copy_if(fBranches.begin(), fBranches.end(), std::back_inserter(branchesWithNullAddress),
                   [](TBranch *b) { return b->GetAddress() == nullptr; });

      if (branchesWithNullAddress.empty())
         return;

      // otherwise build error message and throw
      std::vector<std::string> missingBranchNames;
      std::transform(branchesWithNullAddress.begin(), branchesWithNullAddress.end(),
                     std::back_inserter(missingBranchNames), [](TBranch *b) { return b->GetName(); });
      std::string msg = "RDataFrame::Snapshot:";
      if (missingBranchNames.size() == 1) {
         msg += " branch " + missingBranchNames[0] +
                " is needed as it provides the size for one or more branches containing dynamically sized arrays, but "
                "it is";
      } else {
         msg += " branches ";
         for (const auto &bName : missingBranchNames)
            msg += bName + ", ";
         msg.resize(msg.size() - 2); // remove last ", "
         msg +=
            " are needed as they provide the size of other branches containing dynamically sized arrays, but they are";
      }
      msg += " not part of the set of branches that are being written out.";
      throw std::runtime_error(msg);
   }
};

template <typename T>
void SetBranchesHelper(TTree *inputTree, TTree &outputTree, const std::string &inName, const std::string &name,
                       TBranch *&branch, void *&branchAddress, T *address, RBranchSet &outputBranches,
                       bool /*isDefine*/, int basketSize)
{
   static TClassRef TBOClRef("TBranchObject");

   TBranch *inputBranch = nullptr;
   if (inputTree) {
      inputBranch = inputTree->GetBranch(inName.c_str());
      if (!inputBranch) // try harder
         inputBranch = inputTree->FindBranch(inName.c_str());
   }

   auto *outputBranch = outputBranches.Get(name);
   if (outputBranch) {
      // the output branch was already created, we just need to (re)set its address
      if (inputBranch && inputBranch->IsA() == TBOClRef) {
         outputBranch->SetAddress(reinterpret_cast<T **>(inputBranch->GetAddress()));
      } else if (outputBranch->IsA() != TBranch::Class()) {
         branchAddress = address;
         outputBranch->SetAddress(&branchAddress);
      } else {
         outputBranch->SetAddress(address);
         branchAddress = address;
      }
      return;
   }

   if (inputBranch) {
      // Respect the original bufsize and splitlevel arguments
      // In particular, by keeping splitlevel equal to 0 if this was the case for `inputBranch`, we avoid
      // writing garbage when unsplit objects cannot be written as split objects (e.g. in case of a polymorphic
      // TObject branch, see https://bit.ly/2EjLMId ).
      // A user-provided basket size value takes precedence.
      const auto bufSize = (basketSize > 0) ? basketSize : inputBranch->GetBasketSize();
      const auto splitLevel = inputBranch->GetSplitLevel();

      if (inputBranch->IsA() == TBOClRef) {
         // Need to pass a pointer to pointer
         outputBranch =
            outputTree.Branch(name.c_str(), reinterpret_cast<T **>(inputBranch->GetAddress()), bufSize, splitLevel);
      } else {
         outputBranch = outputTree.Branch(name.c_str(), address, bufSize, splitLevel);
      }
   } else {
      // Set Custom basket size for new branches.
      const auto buffSize = (basketSize > 0) ? basketSize : (inputBranch ? inputBranch->GetBasketSize() : 32000);
      outputBranch = outputTree.Branch(name.c_str(), address, buffSize);
   }
   outputBranches.Insert(name, outputBranch);
   // This is not an array branch, so we don't register the address of the output branch here
   branch = nullptr;
   branchAddress = nullptr;
}

/// Helper function for SnapshotTTreeHelper and SnapshotTTreeHelperMT. It creates new branches for the output TTree of a
/// Snapshot. This overload is called for columns of type `RVec<T>`. For RDF, these can represent:
/// 1. c-style arrays in ROOT files, so we are sure that there are input trees to which we can ask the correct branch
/// title
/// 2. RVecs coming from a custom column or the input file/data-source
/// 3. vectors coming from ROOT files that are being read as RVecs
/// 4. TClonesArray
///
/// In case of 1., we keep aside the pointer to the branch and the pointer to the input value (in `branch` and
/// `branchAddress`) so we can intercept changes in the address of the input branch and tell the output branch.
template <typename T>
void SetBranchesHelper(TTree *inputTree, TTree &outputTree, const std::string &inName, const std::string &outName,
                       TBranch *&branch, void *&branchAddress, RVec<T> *ab, RBranchSet &outputBranches, bool isDefine,
                       int basketSize)
{
   TBranch *inputBranch = nullptr;
   if (inputTree) {
      inputBranch = inputTree->GetBranch(inName.c_str());
      if (!inputBranch) // try harder
         inputBranch = inputTree->FindBranch(inName.c_str());
   }
   auto *outputBranch = outputBranches.Get(outName);

   // if no backing input branch, we must write out an RVec
   bool mustWriteRVec = (inputBranch == nullptr || isDefine);
   // otherwise, if input branch is TClonesArray, must write out an RVec
   if (!mustWriteRVec && std::string_view(inputBranch->GetClassName()) == "TClonesArray") {
      mustWriteRVec = true;
      Warning("Snapshot",
              "Branch \"%s\" contains TClonesArrays but the type specified to Snapshot was RVec<T>. The branch will "
              "be written out as a RVec instead of a TClonesArray. Specify that the type of the branch is "
              "TClonesArray as a Snapshot template parameter to write out a TClonesArray instead.",
              inName.c_str());
   }
   // otherwise, if input branch is a std::vector or RVec, must write out an RVec
   if (!mustWriteRVec) {
      const auto STLKind = TClassEdit::IsSTLCont(inputBranch->GetClassName());
      if (STLKind == ROOT::ESTLType::kSTLvector || STLKind == ROOT::ESTLType::kROOTRVec)
         mustWriteRVec = true;
   }

   if (mustWriteRVec) {
      // Treat:
      // 2. RVec coming from a custom column or a source
      // 3. RVec coming from a column on disk of type vector (the RVec is adopting the data of that vector)
      // 4. TClonesArray written out as RVec<T>
      if (outputBranch) {
         // needs to be SetObject (not SetAddress) to mimic what happens when this TBranchElement is constructed
         outputBranch->SetObject(ab);
      } else {
         // Set Custom basket size for new branches if specified, otherwise get basket size from input branches
         const auto buffSize = (basketSize > 0) ? basketSize : (inputBranch ? inputBranch->GetBasketSize() : 32000);
         auto *b = outputTree.Branch(outName.c_str(), ab, buffSize);
         outputBranches.Insert(outName, b);
      }
      return;
   }

   // else this must be a C-array, aka case 1.
   auto dataPtr = ab->data();

   if (outputBranch) {
      if (outputBranch->IsA() != TBranch::Class()) {
         branchAddress = dataPtr;
         outputBranch->SetAddress(&branchAddress);
      } else {
         outputBranch->SetAddress(dataPtr);
      }
   } else {
      // must construct the leaflist for the output branch and create the branch in the output tree
      auto *const leaf = static_cast<TLeaf *>(inputBranch->GetListOfLeaves()->UncheckedAt(0));
      const auto bname = leaf->GetName();
      auto *sizeLeaf = leaf->GetLeafCount();
      const auto sizeLeafName = sizeLeaf ? std::string(sizeLeaf->GetName()) : std::to_string(leaf->GetLenStatic());

      if (sizeLeaf && !outputBranches.Get(sizeLeafName)) {
         // The output array branch `bname` has dynamic size stored in leaf `sizeLeafName`, but that leaf has not been
         // added to the output tree yet. However, the size leaf has to be available for the creation of the array
         // branch to be successful. So we create the size leaf here.
         const auto sizeTypeStr = TypeName2ROOTTypeName(sizeLeaf->GetTypeName());
         // Use Original basket size for Existing Branches otherwise use Custom basket Size.
         const auto sizeBufSize = (basketSize > 0) ? basketSize : sizeLeaf->GetBranch()->GetBasketSize();
         // The null branch address is a placeholder. It will be set when SetBranchesHelper is called for `sizeLeafName`
         auto *sizeBranch = outputTree.Branch(sizeLeafName.c_str(), (void *)nullptr,
                                              (sizeLeafName + '/' + sizeTypeStr).c_str(), sizeBufSize);
         outputBranches.Insert(sizeLeafName, sizeBranch);
      }

      const auto btype = leaf->GetTypeName();
      const auto rootbtype = TypeName2ROOTTypeName(btype);
      if (rootbtype == ' ') {
         Warning("Snapshot",
                 "RDataFrame::Snapshot: could not correctly construct a leaflist for C-style array in column %s. This "
                 "column will not be written out.",
                 bname);
      } else {
         const auto leaflist = std::string(bname) + "[" + sizeLeafName + "]/" + rootbtype;
         // Use original basket size for existing branches and new basket size for new branches
         const auto branchBufSize = (basketSize > 0) ? basketSize : inputBranch->GetBasketSize();
         outputBranch = outputTree.Branch(outName.c_str(), dataPtr, leaflist.c_str(), branchBufSize);
         outputBranch->SetTitle(inputBranch->GetTitle());
         outputBranches.Insert(outName, outputBranch);
         branch = outputBranch;
         branchAddress = ab->data();
      }
   }
}

void SetBranchesHelper(TTree *inputTree, TTree &outputTree, RBranchSet &outputBranches, int basketSize,
                       const std::string &inputBranchName, const std::string &outputBranchName,
                       const std::type_info &valueTypeID, void *valueAddress, TBranch *&actionHelperBranchPtr,
                       void *&actionHelperBranchPtrAddress);

/// Ensure that the TTree with the resulting snapshot can be written to the target TFile. This means checking that the
/// TFile can be opened in the mode specified in `opts`, deleting any existing TTrees in case
/// `opts.fOverwriteIfExists = true`, or throwing an error otherwise.
void EnsureValidSnapshotTTreeOutput(const RSnapshotOptions &opts, const std::string &treeName,
                                    const std::string &fileName);

/// Helper object for a single-thread TTree-based Snapshot action
template <typename... ColTypes>
class R__CLING_PTRCHECK(off) SnapshotTTreeHelper : public RActionImpl<SnapshotTTreeHelper<ColTypes...>> {
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

public:
   using ColumnTypes_t = TypeList<ColTypes...>;
   SnapshotTTreeHelper(std::string_view filename, std::string_view dirname, std::string_view treename,
                       const ColumnNames_t &vbnames, const ColumnNames_t &bnames, const RSnapshotOptions &options,
                       std::vector<bool> &&isDefine, ROOT::Detail::RDF::RLoopManager *loopManager,
                       ROOT::Detail::RDF::RLoopManager *inputLM)
      : fFileName(filename),
        fDirName(dirname),
        fTreeName(treename),
        fOptions(options),
        fInputBranchNames(vbnames),
        fOutputBranchNames(ReplaceDotWithUnderscore(bnames)),
        fBranches(vbnames.size(), nullptr),
        fBranchAddresses(vbnames.size(), nullptr),
        fIsDefine(std::move(isDefine)),
        fOutputLoopManager(loopManager),
        fInputLoopManager(inputLM)
   {
      EnsureValidSnapshotTTreeOutput(fOptions, fTreeName, fFileName);
   }

   SnapshotTTreeHelper(const SnapshotTTreeHelper &) = delete;
   SnapshotTTreeHelper(SnapshotTTreeHelper &&) = default;
   ~SnapshotTTreeHelper()
   {
      if (!fTreeName.empty() /*not moved from*/ && !fOutputFile /* did not run */ && fOptions.fLazy) {
         const auto fileOpenMode = [&]() {
            TString checkupdate = fOptions.fMode;
            checkupdate.ToLower();
            return checkupdate == "update" ? "updated" : "created";
         }();
         Warning("Snapshot",
                 "A lazy Snapshot action was booked but never triggered. The tree '%s' in output file '%s' was not %s. "
                 "In case it was desired instead, remember to trigger the Snapshot operation, by storing "
                 "its result in a variable and for example calling the GetValue() method on it.",
                 fTreeName.c_str(), fFileName.c_str(), fileOpenMode);
      }
   }

   void InitTask(TTreeReader * /*treeReader*/, unsigned int /* slot */)
   {
      // We ask the input RLoopManager if it has a TTree. We cannot rely on getting this information when constructing
      // this action helper, since the TTree might change e.g. when ChangeSpec is called in-between distributed tasks.
      fInputTree = fInputLoopManager->GetTree();
      fBranchAddressesNeedReset = true;
   }

   void Exec(unsigned int /* slot */, ColTypes &...values)
   {
      using ind_t = std::index_sequence_for<ColTypes...>;
      if (!fBranchAddressesNeedReset) {
         UpdateCArraysPtrs(values..., ind_t{});
      } else {
         SetBranches(values..., ind_t{});
         fBranchAddressesNeedReset = false;
      }
      fOutputTree->Fill();
   }

   template <std::size_t... S>
   void UpdateCArraysPtrs(ColTypes &...values, std::index_sequence<S...> /*dummy*/)
   {
      // This code deals with branches which hold C arrays of variable size. It can happen that the buffers
      // associated to those is re-allocated. As a result the value of the pointer can change therewith
      // leaving associated to the branch of the output tree an invalid pointer.
      // With this code, we set the value of the pointer in the output branch anew when needed.
      // Nota bene: the extra ",0" after the invocation of SetAddress, is because that method returns void and
      // we need an int for the expander list.
      int expander[] = {(fBranches[S] && fBranchAddresses[S] != GetData(values)
                         ? fBranches[S]->SetAddress(GetData(values)),
                         fBranchAddresses[S] = GetData(values), 0 : 0, 0)...,
                        0};
      (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
   }

   template <std::size_t... S>
   void SetBranches(ColTypes &...values, std::index_sequence<S...> /*dummy*/)
   {
      // create branches in output tree
      int expander[] = {
         (SetBranchesHelper(fInputTree, *fOutputTree, fInputBranchNames[S], fOutputBranchNames[S], fBranches[S],
                            fBranchAddresses[S], &values, fOutputBranches, fIsDefine[S], fOptions.fBasketSize),
          0)...,
         0};
      fOutputBranches.AssertNoNullBranchAddresses();
      (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
   }

   template <std::size_t... S>
   void SetEmptyBranches(TTree *inputTree, TTree &outputTree, std::index_sequence<S...>)
   {
      RBranchSet outputBranches{};
      void *dummyValueAddress{};
      TBranch *dummyTBranchPtr{};
      void *dummyTBranchAddress{};
      // We use the expander trick rather than a fold expression to avoid incurring in the bracket depth limit of clang
      int expander[] = {(SetBranchesHelper(inputTree, outputTree, outputBranches, fOptions.fBasketSize,
                                           fInputBranchNames[S], fOutputBranchNames[S], typeid(ColTypes),
                                           dummyValueAddress, dummyTBranchPtr, dummyTBranchAddress),
                         0)...,
                        0};
      (void)expander;
   }

   void Initialize()
   {
      fOutputFile.reset(
         TFile::Open(fFileName.c_str(), fOptions.fMode.c_str(), /*ftitle=*/"",
                     ROOT::CompressionSettings(fOptions.fCompressionAlgorithm, fOptions.fCompressionLevel)));
      if (!fOutputFile)
         throw std::runtime_error("Snapshot: could not create output file " + fFileName);

      TDirectory *outputDir = fOutputFile.get();
      if (!fDirName.empty()) {
         TString checkupdate = fOptions.fMode;
         checkupdate.ToLower();
         if (checkupdate == "update")
            outputDir = fOutputFile->mkdir(fDirName.c_str(), "", true); // do not overwrite existing directory
         else
            outputDir = fOutputFile->mkdir(fDirName.c_str());
      }

      fOutputTree =
         std::make_unique<TTree>(fTreeName.c_str(), fTreeName.c_str(), fOptions.fSplitLevel, /*dir=*/outputDir);

      if (fOptions.fAutoFlush)
         fOutputTree->SetAutoFlush(fOptions.fAutoFlush);
   }

   void Finalize()
   {
      assert(fOutputTree != nullptr);
      assert(fOutputFile != nullptr);

      // There were no entries to fill the TTree with (either the input TTree was empty or no event passed after
      // filtering). We have already created an empty TTree, now also create the branches to preserve the schema
      if (fOutputTree->GetEntries() == 0) {
         using ind_t = std::index_sequence_for<ColTypes...>;
         SetEmptyBranches(fInputTree, *fOutputTree, ind_t{});
      }
      // use AutoSave to flush TTree contents because TTree::Write writes in gDirectory, not in fDirectory
      fOutputTree->AutoSave("flushbaskets");
      // must destroy the TTree first, otherwise TFile will delete it too leading to a double delete
      fOutputTree.reset();
      fOutputFile->Close();

      // Now connect the data source to the loop manager so it can be used for further processing
      auto fullTreeName = fDirName.empty() ? fTreeName : fDirName + '/' + fTreeName;
      fOutputLoopManager->SetDataSource(std::make_unique<ROOT::Internal::RDF::RTTreeDS>(fullTreeName, fFileName));
   }

   std::string GetActionName() { return "Snapshot"; }

   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return [this](unsigned int, const RSampleInfo &) mutable { fBranchAddressesNeedReset = true; };
   }

   /**
    * @brief Create a new SnapshotTTreeHelper with a different output file name
    *
    * @param newName A type-erased string with the output file name
    * @return SnapshotTTreeHelper
    *
    * This MakeNew implementation is tied to the cloning feature of actions
    * of the computation graph. In particular, cloning a Snapshot node usually
    * also involves changing the name of the output file, otherwise the cloned
    * Snapshot would overwrite the same file.
    */
   SnapshotTTreeHelper MakeNew(void *newName, std::string_view /*variation*/ = "nominal")
   {
      const std::string finalName = *reinterpret_cast<const std::string *>(newName);
      return SnapshotTTreeHelper{finalName,
                                 fDirName,
                                 fTreeName,
                                 fInputBranchNames,
                                 fOutputBranchNames,
                                 fOptions,
                                 std::vector<bool>(fIsDefine),
                                 fOutputLoopManager,
                                 fInputLoopManager};
   }
};

/// Helper object for a multi-thread TTree-based Snapshot action
template <typename... ColTypes>
class R__CLING_PTRCHECK(off) SnapshotTTreeHelperMT : public RActionImpl<SnapshotTTreeHelperMT<ColTypes...>> {
   unsigned int fNSlots;
   std::unique_ptr<ROOT::TBufferMerger> fMerger; // must use a ptr because TBufferMerger is not movable
   std::vector<std::shared_ptr<ROOT::TBufferMergerFile>> fOutputFiles;
   std::vector<std::unique_ptr<TTree>> fOutputTrees;
   std::vector<int> fBranchAddressesNeedReset; // vector<bool> does not allow concurrent writing of different elements
   std::string fFileName;                      // name of the output file name
   std::string fDirName;            // name of TFile subdirectory in which output must be written (possibly empty)
   std::string fTreeName;           // name of output tree
   RSnapshotOptions fOptions;       // struct holding options to pass down to TFile and TTree in this action
   ColumnNames_t fInputBranchNames; // This contains the resolved aliases
   ColumnNames_t fOutputBranchNames;
   std::vector<TTree *> fInputTrees; // Current input trees. Set at initialization time (`InitTask`)
   // Addresses of branches in output per slot, non-null only for the ones holding C arrays
   std::vector<std::vector<TBranch *>> fBranches;
   // Addresses associated to output branches per slot, non-null only for the ones holding C arrays
   std::vector<std::vector<void *>> fBranchAddresses;
   std::vector<RBranchSet> fOutputBranches;
   std::vector<bool> fIsDefine;
   ROOT::Detail::RDF::RLoopManager *fOutputLoopManager;
   ROOT::Detail::RDF::RLoopManager *fInputLoopManager;
   TFile *fOutputFile; // Non-owning view on the output file

public:
   using ColumnTypes_t = TypeList<ColTypes...>;

   SnapshotTTreeHelperMT(const unsigned int nSlots, std::string_view filename, std::string_view dirname,
                         std::string_view treename, const ColumnNames_t &vbnames, const ColumnNames_t &bnames,
                         const RSnapshotOptions &options, std::vector<bool> &&isDefine,
                         ROOT::Detail::RDF::RLoopManager *loopManager, ROOT::Detail::RDF::RLoopManager *inputLM)
      : fNSlots(nSlots),
        fOutputFiles(fNSlots),
        fOutputTrees(fNSlots),
        fBranchAddressesNeedReset(fNSlots, 1),
        fFileName(filename),
        fDirName(dirname),
        fTreeName(treename),
        fOptions(options),
        fInputBranchNames(vbnames),
        fOutputBranchNames(ReplaceDotWithUnderscore(bnames)),
        fInputTrees(fNSlots),
        fBranches(fNSlots, std::vector<TBranch *>(vbnames.size(), nullptr)),
        fBranchAddresses(fNSlots, std::vector<void *>(vbnames.size(), nullptr)),
        fOutputBranches(fNSlots),
        fIsDefine(std::move(isDefine)),
        fOutputLoopManager(loopManager),
        fInputLoopManager(inputLM)
   {
      EnsureValidSnapshotTTreeOutput(fOptions, fTreeName, fFileName);
   }
   SnapshotTTreeHelperMT(const SnapshotTTreeHelperMT &) = delete;
   SnapshotTTreeHelperMT(SnapshotTTreeHelperMT &&) = default;
   ~SnapshotTTreeHelperMT()
   {
      if (!fTreeName.empty() /*not moved from*/ && fOptions.fLazy && !fOutputFiles.empty() &&
          std::all_of(fOutputFiles.begin(), fOutputFiles.end(), [](const auto &f) { return !f; }) /* never run */) {
         const auto fileOpenMode = [&]() {
            TString checkupdate = fOptions.fMode;
            checkupdate.ToLower();
            return checkupdate == "update" ? "updated" : "created";
         }();
         Warning("Snapshot",
                 "A lazy Snapshot action was booked but never triggered. The tree '%s' in output file '%s' was not %s. "
                 "In case it was desired instead, remember to trigger the Snapshot operation, by storing "
                 "its result in a variable and for example calling the GetValue() method on it.",
                 fTreeName.c_str(), fFileName.c_str(), fileOpenMode);
      }
   }

   void InitTask(TTreeReader *r, unsigned int slot)
   {
      ::TDirectory::TContext c; // do not let tasks change the thread-local gDirectory
      if (!fOutputFiles[slot]) {
         // first time this thread executes something, let's create a TBufferMerger output directory
         fOutputFiles[slot] = fMerger->GetFile();
      }
      TDirectory *treeDirectory = fOutputFiles[slot].get();
      if (!fDirName.empty()) {
         // call returnExistingDirectory=true since MT can end up making this call multiple times
         treeDirectory = fOutputFiles[slot]->mkdir(fDirName.c_str(), "", true);
      }
      // re-create output tree as we need to create its branches again, with new input variables
      // TODO we could instead create the output tree and its branches, change addresses of input variables in each task
      fOutputTrees[slot] =
         std::make_unique<TTree>(fTreeName.c_str(), fTreeName.c_str(), fOptions.fSplitLevel, /*dir=*/treeDirectory);
      fOutputTrees[slot]->SetBit(TTree::kEntriesReshuffled);
      // TODO can be removed when RDF supports interleaved TBB task execution properly, see ROOT-10269
      fOutputTrees[slot]->SetImplicitMT(false);
      if (fOptions.fAutoFlush)
         fOutputTrees[slot]->SetAutoFlush(fOptions.fAutoFlush);
      if (r) {
         // We could be getting a task-local TTreeReader from the TTreeProcessorMT.
         fInputTrees[slot] = r->GetTree();
      } else {
         fInputTrees[slot] = fInputLoopManager->GetTree();
      }
      fBranchAddressesNeedReset[slot] = 1; // reset first event flag for this slot
   }

   void FinalizeTask(unsigned int slot)
   {
      if (fOutputTrees[slot]->GetEntries() > 0)
         fOutputFiles[slot]->Write();
      // clear now to avoid concurrent destruction of output trees and input tree (which has them listed as fClones)
      fOutputTrees[slot].reset(nullptr);
      fOutputBranches[slot].Clear();
   }

   void Exec(unsigned int slot, ColTypes &...values)
   {
      using ind_t = std::index_sequence_for<ColTypes...>;
      if (fBranchAddressesNeedReset[slot] == 0) {
         UpdateCArraysPtrs(slot, values..., ind_t{});
      } else {
         SetBranches(slot, values..., ind_t{});
         fBranchAddressesNeedReset[slot] = 0;
      }
      fOutputTrees[slot]->Fill();
      auto entries = fOutputTrees[slot]->GetEntries();
      auto autoFlush = fOutputTrees[slot]->GetAutoFlush();
      if ((autoFlush > 0) && (entries % autoFlush == 0))
         fOutputFiles[slot]->Write();
   }

   template <std::size_t... S>
   void UpdateCArraysPtrs(unsigned int slot, ColTypes &...values, std::index_sequence<S...> /*dummy*/)
   {
      // This code deals with branches which hold C arrays of variable size. It can happen that the buffers
      // associated to those is re-allocated. As a result the value of the pointer can change therewith
      // leaving associated to the branch of the output tree an invalid pointer.
      // With this code, we set the value of the pointer in the output branch anew when needed.
      // Nota bene: the extra ",0" after the invocation of SetAddress, is because that method returns void and
      // we need an int for the expander list.
      int expander[] = {(fBranches[slot][S] && fBranchAddresses[slot][S] != GetData(values)
                         ? fBranches[slot][S]->SetAddress(GetData(values)),
                         fBranchAddresses[slot][S] = GetData(values), 0 : 0, 0)...,
                        0};
      (void)expander; // avoid unused parameter warnings (gcc 12.1)
      (void)slot;     // Also "slot" might be unused, in case "values" is empty
   }

   template <std::size_t... S>
   void SetBranches(unsigned int slot, ColTypes &...values, std::index_sequence<S...> /*dummy*/)
   {
      // hack to call TTree::Branch on all variadic template arguments
      int expander[] = {(SetBranchesHelper(fInputTrees[slot], *fOutputTrees[slot], fInputBranchNames[S],
                                           fOutputBranchNames[S], fBranches[slot][S], fBranchAddresses[slot][S],
                                           &values, fOutputBranches[slot], fIsDefine[S], fOptions.fBasketSize),
                         0)...,
                        0};
      fOutputBranches[slot].AssertNoNullBranchAddresses();
      (void)expander; // avoid unused parameter warnings (gcc 12.1)
   }

   template <std::size_t... S>
   void SetEmptyBranches(TTree *inputTree, TTree &outputTree, std::index_sequence<S...>)
   {
      void *dummyValueAddress{};
      TBranch *dummyTBranchPtr{};
      void *dummyTBranchAddress{};
      RBranchSet outputBranches{};
      // We use the expander trick rather than a fold expression to avoid incurring in the bracket depth limit of clang
      int expander[] = {(SetBranchesHelper(inputTree, outputTree, outputBranches, fOptions.fBasketSize,
                                           fInputBranchNames[S], fOutputBranchNames[S], typeid(ColTypes),
                                           dummyValueAddress, dummyTBranchPtr, dummyTBranchAddress),
                         0)...,
                        0};
      (void)expander;
   }

   void Initialize()
   {
      const auto cs = ROOT::CompressionSettings(fOptions.fCompressionAlgorithm, fOptions.fCompressionLevel);
      auto outFile = std::unique_ptr<TFile>{
         TFile::Open(fFileName.c_str(), fOptions.fMode.c_str(), /*ftitle=*/fFileName.c_str(), cs)};
      if (!outFile)
         throw std::runtime_error("Snapshot: could not create output file " + fFileName);
      fOutputFile = outFile.get();
      fMerger = std::make_unique<ROOT::TBufferMerger>(std::move(outFile));
   }

   void Finalize()
   {
      assert(std::any_of(fOutputFiles.begin(), fOutputFiles.end(), [](const auto &ptr) { return ptr != nullptr; }));

      for (auto &file : fOutputFiles) {
         if (file) {
            file->Write();
            file->Close();
         }
      }

      // If there were no entries to fill the TTree with (either the input TTree was empty or no event passed after
      // filtering), create an empty TTree in the output file and create the branches to preserve the schema
      auto fullTreeName = fDirName.empty() ? fTreeName : fDirName + '/' + fTreeName;
      assert(fOutputFile && "Missing output file in Snapshot finalization.");
      if (!fOutputFile->Get(fullTreeName.c_str())) {

         // First find in which directory we need to write the output TTree
         TDirectory *treeDirectory = fOutputFile;
         if (!fDirName.empty()) {
            treeDirectory = fOutputFile->mkdir(fDirName.c_str(), "", true);
         }
         ::TDirectory::TContext c{treeDirectory};

         // Create the output TTree and create the user-requested branches
         auto outTree =
            std::make_unique<TTree>(fTreeName.c_str(), fTreeName.c_str(), fOptions.fSplitLevel, /*dir=*/treeDirectory);
         using ind_t = std::index_sequence_for<ColTypes...>;
         SetEmptyBranches(fInputLoopManager->GetTree(), *outTree, ind_t{});

         fOutputFile->Write();
      }

      // flush all buffers to disk by destroying the TBufferMerger
      fOutputFiles.clear();
      fMerger.reset();

      // Now connect the data source to the loop manager so it can be used for further processing
      fOutputLoopManager->SetDataSource(std::make_unique<ROOT::Internal::RDF::RTTreeDS>(fullTreeName, fFileName));
   }

   std::string GetActionName() { return "Snapshot"; }

   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return [this](unsigned int slot, const RSampleInfo &) mutable { fBranchAddressesNeedReset[slot] = 1; };
   }

   /**
    * @brief Create a new SnapshotTTreeHelperMT with a different output file name
    *
    * @param newName A type-erased string with the output file name
    * @return SnapshotTTreeHelperMT
    *
    * This MakeNew implementation is tied to the cloning feature of actions
    * of the computation graph. In particular, cloning a Snapshot node usually
    * also involves changing the name of the output file, otherwise the cloned
    * Snapshot would overwrite the same file.
    */
   SnapshotTTreeHelperMT MakeNew(void *newName, std::string_view /*variation*/ = "nominal")
   {
      const std::string finalName = *reinterpret_cast<const std::string *>(newName);
      return SnapshotTTreeHelperMT{fNSlots,
                                   finalName,
                                   fDirName,
                                   fTreeName,
                                   fInputBranchNames,
                                   fOutputBranchNames,
                                   fOptions,
                                   std::vector<bool>(fIsDefine),
                                   fOutputLoopManager,
                                   fInputLoopManager};
   }
};

/// Ensure that the RNTuple with the resulting snapshot can be written to the target TFile. This means checking that the
/// TFile can be opened in the mode specified in `opts`, deleting any existing RNTuples in case
/// `opts.fOverwriteIfExists = true`, or throwing an error otherwise.
void EnsureValidSnapshotRNTupleOutput(const RSnapshotOptions &opts, const std::string &ntupleName,
                                      const std::string &fileName);

/// Helper function to update the value of an RNTuple's field in the provided entry.
template <typename T>
void SetFieldsHelper(T &value, std::string_view fieldName, ROOT::REntry *entry)
{
   entry->BindRawPtr(fieldName, &value);
}

/// Helper object for a single-thread RNTuple-based Snapshot action
template <typename... ColTypes>
class R__CLING_PTRCHECK(off) SnapshotRNTupleHelper : public RActionImpl<SnapshotRNTupleHelper<ColTypes...>> {
   std::string fFileName;
   std::string fDirName;
   std::string fNTupleName;

   std::unique_ptr<TFile> fOutputFile{nullptr};

   RSnapshotOptions fOptions;
   ROOT::Detail::RDF::RLoopManager *fOutputLoopManager;
   ColumnNames_t fInputFieldNames; // This contains the resolved aliases
   ColumnNames_t fOutputFieldNames;
   std::unique_ptr<ROOT::RNTupleWriter> fWriter{nullptr};

   ROOT::REntry *fOutputEntry;

   std::vector<bool> fIsDefine;

public:
   using ColumnTypes_t = TypeList<ColTypes...>;
   SnapshotRNTupleHelper(std::string_view filename, std::string_view dirname, std::string_view ntuplename,
                         const ColumnNames_t &vfnames, const ColumnNames_t &fnames, const RSnapshotOptions &options,
                         ROOT::Detail::RDF::RLoopManager *lm, std::vector<bool> &&isDefine)
      : fFileName(filename),
        fDirName(dirname),
        fNTupleName(ntuplename),
        fOptions(options),
        fOutputLoopManager(lm),
        fInputFieldNames(vfnames),
        fOutputFieldNames(ReplaceDotWithUnderscore(fnames)),
        fIsDefine(std::move(isDefine))
   {
      EnsureValidSnapshotRNTupleOutput(fOptions, fNTupleName, fFileName);
   }

   SnapshotRNTupleHelper(const SnapshotRNTupleHelper &) = delete;
   SnapshotRNTupleHelper &operator=(const SnapshotRNTupleHelper &) = delete;
   SnapshotRNTupleHelper(SnapshotRNTupleHelper &&) = default;
   SnapshotRNTupleHelper &operator=(SnapshotRNTupleHelper &&) = default;
   ~SnapshotRNTupleHelper()
   {
      if (!fNTupleName.empty() && !fOutputLoopManager->GetDataSource() && fOptions.fLazy)
         Warning("Snapshot", "A lazy Snapshot action was booked but never triggered.");
   }

   void InitTask(TTreeReader *, unsigned int /* slot */) {}

   void Exec(unsigned int /* slot */, ColTypes &...values)
   {
      using ind_t = std::index_sequence_for<ColTypes...>;

      SetFields(values..., ind_t{});
      fWriter->Fill();
   }

   template <std::size_t... S>
   void SetFields(ColTypes &...values, std::index_sequence<S...> /*dummy*/)
   {
      int expander[] = {(SetFieldsHelper(values, fOutputFieldNames[S], fOutputEntry), 0)..., 0};
      (void)expander; // avoid unused variable warnings for older compilers (gcc 14.1)
   }

   void Initialize()
   {
      using ind_t = std::index_sequence_for<ColTypes...>;

      auto model = ROOT::RNTupleModel::Create();
      MakeFields(*model, ind_t{});
      fOutputEntry = &model->GetDefaultEntry();

      ROOT::RNTupleWriteOptions writeOptions;
      writeOptions.SetCompression(fOptions.fCompressionAlgorithm, fOptions.fCompressionLevel);

      fOutputFile.reset(TFile::Open(fFileName.c_str(), fOptions.fMode.c_str()));
      if (!fOutputFile)
         throw std::runtime_error("Snapshot: could not create output file " + fFileName);

      TDirectory *outputDir = fOutputFile.get();
      if (!fDirName.empty()) {
         TString checkupdate = fOptions.fMode;
         checkupdate.ToLower();
         if (checkupdate == "update")
            outputDir = fOutputFile->mkdir(fDirName.c_str(), "", true); // do not overwrite existing directory
         else
            outputDir = fOutputFile->mkdir(fDirName.c_str());
      }

      fWriter = ROOT::RNTupleWriter::Append(std::move(model), fNTupleName, *outputDir, writeOptions);
   }

   template <std::size_t... S>
   void MakeFields(ROOT::RNTupleModel &model, std::index_sequence<S...> /*dummy*/)
   {
      int expander[] = {(model.MakeField<ColTypes>(fOutputFieldNames[S]), 0)..., 0};
      (void)expander; // avoid unused variable warnings for older compilers (gcc 14.1)
   }

   void Finalize()
   {
      fWriter.reset();
      // We can now set the data source of the loop manager for the RDataFrame that is returned by the Snapshot call.
      fOutputLoopManager->SetDataSource(
         std::make_unique<ROOT::RDF::RNTupleDS>(fDirName + "/" + fNTupleName, fFileName));
   }

   std::string GetActionName() { return "Snapshot"; }

   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return [](unsigned int, const RSampleInfo &) mutable {};
   }

   /**
    * @brief Create a new SnapshotRNTupleHelper with a different output file name
    *
    * @param newName A type-erased string with the output file name
    * @return SnapshotRNTupleHelper
    *
    * This MakeNew implementation is tied to the cloning feature of actions
    * of the computation graph. In particular, cloning a Snapshot node usually
    * also involves changing the name of the output file, otherwise the cloned
    * Snapshot would overwrite the same file.
    */
   SnapshotRNTupleHelper MakeNew(void *newName)
   {
      const std::string finalName = *reinterpret_cast<const std::string *>(newName);
      return SnapshotRNTupleHelper{finalName,
                                   fNTupleName,
                                   fInputFieldNames,
                                   fOutputFieldNames,
                                   fOptions,
                                   fOutputLoopManager,
                                   std::vector<bool>(fIsDefine)};
   }
};

class R__CLING_PTRCHECK(off) UntypedSnapshotRNTupleHelper final : public RActionImpl<UntypedSnapshotRNTupleHelper> {
   std::string fFileName;
   std::string fDirName;
   std::string fNTupleName;

   std::unique_ptr<TFile> fOutputFile{nullptr};

   RSnapshotOptions fOptions;
   ROOT::Detail::RDF::RLoopManager *fInputLoopManager;
   ROOT::Detail::RDF::RLoopManager *fOutputLoopManager;
   ColumnNames_t fInputFieldNames; // This contains the resolved aliases
   ColumnNames_t fOutputFieldNames;
   std::unique_ptr<ROOT::RNTupleWriter> fWriter{nullptr};

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
   UntypedSnapshotRNTupleHelper(UntypedSnapshotRNTupleHelper &&) = default;
   UntypedSnapshotRNTupleHelper &operator=(UntypedSnapshotRNTupleHelper &&) = default;
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
   UntypedSnapshotTTreeHelper(UntypedSnapshotTTreeHelper &&) = default;
   UntypedSnapshotTTreeHelper &operator=(UntypedSnapshotTTreeHelper &&) = default;
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
   UntypedSnapshotTTreeHelperMT(UntypedSnapshotTTreeHelperMT &&) = default;
   UntypedSnapshotTTreeHelperMT &operator=(UntypedSnapshotTTreeHelperMT &&) = default;
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