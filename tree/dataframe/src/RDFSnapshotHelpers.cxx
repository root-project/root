/**
 \file RDFSnapshotHelpers.cxx
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

#include <ROOT/RDF/SnapshotHelpers.hxx>

#include <ROOT/REntry.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RTTreeDS.hxx>
#include <ROOT/TBufferMerger.hxx>

#include <TBranchObject.h>
#include <TClassEdit.h>
#include <TDictionary.h>
#include <TDataType.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TTreeReader.h>

namespace {

TBranch *SearchForBranch(TTree *inputTree, const std::string &branchName)
{
   if (inputTree) {
      if (auto *getBranchRes = inputTree->GetBranch(branchName.c_str()))
         return getBranchRes;

      // try harder
      if (auto *findBranchRes = inputTree->FindBranch(branchName.c_str()))
         return findBranchRes;
   }
   return nullptr;
}

void CreateCStyleArrayBranch(TTree &outputTree, ROOT::Internal::RDF::RBranchSet &outputBranches, TBranch *inputBranch,
                             const std::string &outputBranchName, int basketSize, void *address)
{
   if (!inputBranch)
      return;
   const auto STLKind = TClassEdit::IsSTLCont(inputBranch->GetClassName());
   if (STLKind == ROOT::ESTLType::kSTLvector || STLKind == ROOT::ESTLType::kROOTRVec)
      return;
   // must construct the leaflist for the output branch and create the branch in the output tree
   const auto *leaf = static_cast<TLeaf *>(inputBranch->GetListOfLeaves()->UncheckedAt(0));
   if (!leaf)
      return;
   const auto bname = leaf->GetName();
   auto *sizeLeaf = leaf->GetLeafCount();
   const auto sizeLeafName = sizeLeaf ? std::string(sizeLeaf->GetName()) : std::to_string(leaf->GetLenStatic());

   // We proceed only if branch is a fixed-or-variable-sized array
   if (sizeLeaf || leaf->GetLenStatic() > 1) {
      if (sizeLeaf && !outputBranches.Get(sizeLeafName)) {
         // The output array branch `bname` has dynamic size stored in leaf `sizeLeafName`, but that leaf has not been
         // added to the output tree yet. However, the size leaf has to be available for the creation of the array
         // branch to be successful. So we create the size leaf here.
         const auto sizeTypeStr = ROOT::Internal::RDF::TypeName2ROOTTypeName(sizeLeaf->GetTypeName());
         // Use Original basket size for Existing Branches otherwise use Custom basket Size.
         const auto bufSize = (basketSize > 0) ? basketSize : sizeLeaf->GetBranch()->GetBasketSize();
         // The null branch address is a placeholder. It will be set when SetBranchesHelper is called for `sizeLeafName`
         auto *outputBranch = outputTree.Branch(sizeLeafName.c_str(), static_cast<void *>(nullptr),
                                                (sizeLeafName + '/' + sizeTypeStr).c_str(), bufSize);
         outputBranches.Insert(sizeLeafName, outputBranch);
      }

      const auto btype = leaf->GetTypeName();
      const auto rootbtype = ROOT::Internal::RDF::TypeName2ROOTTypeName(btype);
      if (rootbtype == ' ') {
         Warning("Snapshot",
                 "RDataFrame::Snapshot: could not correctly construct a leaflist for C-style array in column %s. The "
                 "leaf is of type '%s'. This column will not be written out.",
                 bname, btype);
         return;
      }

      const auto leaflist = std::string(bname) + "[" + sizeLeafName + "]/" + rootbtype;
      // Use original basket size for existing branches and new basket size for new branches
      const auto bufSize = (basketSize > 0) ? basketSize : inputBranch->GetBasketSize();
      void *addressForBranch = [address]() -> void * {
         if (address) {
            // Address here points to a ROOT::RVec<std::byte> coming from RTreeUntypedArrayColumnReader. We know we need
            // its buffer, so we cast it and extract the address of the buffer
            auto *rawRVec = reinterpret_cast<ROOT::RVec<std::byte> *>(address);
            return rawRVec->data();
         }
         return nullptr;
      }();
      auto *outputBranch = outputTree.Branch(outputBranchName.c_str(), addressForBranch, leaflist.c_str(), bufSize);
      outputBranch->SetTitle(inputBranch->GetTitle());
      outputBranches.Insert(outputBranchName, outputBranch, true);
   }
}

void SetBranchAddress(TBranch *inputBranch, TBranch &outputBranch, void *&outputBranchAddress, bool isCArray,
                      void *valueAddress)
{
   const static TClassRef TBOClRef("TBranchObject");
   if (inputBranch && inputBranch->IsA() == TBOClRef) {
      outputBranch.SetAddress(reinterpret_cast<void **>(inputBranch->GetAddress()));
   } else if (outputBranch.IsA() != TBranch::Class()) {
      outputBranchAddress = valueAddress;
      outputBranch.SetAddress(&outputBranchAddress);
   } else {
      void *correctAddress = [valueAddress, isCArray]() -> void * {
         if (isCArray) {
            // Address here points to a ROOT::RVec<std::byte> coming from RTreeUntypedArrayColumnReader. We know we
            // need its buffer, so we cast it and extract the address of the buffer
            auto *rawRVec = reinterpret_cast<ROOT::RVec<std::byte> *>(valueAddress);
            return rawRVec->data();
         }
         return valueAddress;
      }();
      outputBranch.SetAddress(correctAddress);
      outputBranchAddress = valueAddress;
   }
}

void CreateFundamentalTypeBranch(TTree &outputTree, const std::string &outputBranchName, void *valueAddress,
                                 const std::type_info &valueTypeID, ROOT::Internal::RDF::RBranchSet &outputBranches,
                                 int bufSize)
{
   // Logic taken from
   // TTree::BranchImpRef(
   // const char* branchname, TClass* ptrClass, EDataType datatype, void* addobj, Int_t bufsize, Int_t splitlevel)
   auto rootTypeChar = ROOT::Internal::RDF::TypeID2ROOTTypeName(valueTypeID);
   if (rootTypeChar == ' ') {
      Warning("Snapshot",
              "RDataFrame::Snapshot: could not correctly construct a leaflist for fundamental type in column %s. This "
              "column will not be written out.",
              outputBranchName.c_str());
      return;
   }
   std::string leafList{outputBranchName + '/' + rootTypeChar};
   auto *outputBranch = outputTree.Branch(outputBranchName.c_str(), valueAddress, leafList.c_str(), bufSize);
   outputBranches.Insert(outputBranchName, outputBranch);
}

/// Ensure that the TTree with the resulting snapshot can be written to the target TFile. This means checking that the
/// TFile can be opened in the mode specified in `opts`, deleting any existing TTrees in case
/// `opts.fOverwriteIfExists = true`, or throwing an error otherwise.
void EnsureValidSnapshotTTreeOutput(const ROOT::RDF::RSnapshotOptions &opts, const std::string &treeName,
                                    const std::string &fileName)
{
   TString fileMode = opts.fMode;
   fileMode.ToLower();
   if (fileMode != "update")
      return;

   // output file opened in "update" mode: must check whether output TTree is already present in file
   std::unique_ptr<TFile> outFile{TFile::Open(fileName.c_str(), "update")};
   if (!outFile || outFile->IsZombie())
      throw std::invalid_argument("Snapshot: cannot open file \"" + fileName + "\" in update mode");

   TObject *outTree = outFile->Get(treeName.c_str());
   if (outTree == nullptr)
      return;

   // object called treeName is already present in the file
   if (opts.fOverwriteIfExists) {
      if (outTree->InheritsFrom("TTree")) {
         static_cast<TTree *>(outTree)->Delete("all");
      } else {
         outFile->Delete(treeName.c_str());
      }
   } else {
      const std::string msg = "Snapshot: tree \"" + treeName + "\" already present in file \"" + fileName +
                              "\". If you want to delete the original tree and write another, please set "
                              "RSnapshotOptions::fOverwriteIfExists to true.";
      throw std::invalid_argument(msg);
   }
}

/// Ensure that the RNTuple with the resulting snapshot can be written to the target TFile. This means checking that the
/// TFile can be opened in the mode specified in `opts`, deleting any existing RNTuples in case
/// `opts.fOverwriteIfExists = true`, or throwing an error otherwise.
void EnsureValidSnapshotRNTupleOutput(const ROOT::RDF::RSnapshotOptions &opts, const std::string &ntupleName,
                                      const std::string &fileName)
{
   TString fileMode = opts.fMode;
   fileMode.ToLower();
   if (fileMode != "update")
      return;

   // output file opened in "update" mode: must check whether output RNTuple is already present in file
   std::unique_ptr<TFile> outFile{TFile::Open(fileName.c_str(), "update")};
   if (!outFile || outFile->IsZombie())
      throw std::invalid_argument("Snapshot: cannot open file \"" + fileName + "\" in update mode");

   auto *outNTuple = outFile->Get<ROOT::RNTuple>(ntupleName.c_str());

   if (outNTuple) {
      if (opts.fOverwriteIfExists) {
         outFile->Delete((ntupleName + ";*").c_str());
         return;
      } else {
         const std::string msg = "Snapshot: RNTuple \"" + ntupleName + "\" already present in file \"" + fileName +
                                 "\". If you want to delete the original ntuple and write another, please set "
                                 "the 'fOverwriteIfExists' option to true in RSnapshotOptions.";
         throw std::invalid_argument(msg);
      }
   }

   // Also check if there is any object other than an RNTuple with the provided ntupleName.
   TObject *outObj = outFile->Get(ntupleName.c_str());

   if (!outObj)
      return;

   // An object called ntupleName is already present in the file.
   if (opts.fOverwriteIfExists) {
      if (auto tree = dynamic_cast<TTree *>(outObj)) {
         tree->Delete("all");
      } else {
         outFile->Delete((ntupleName + ";*").c_str());
      }
   } else {
      const std::string msg = "Snapshot: object \"" + ntupleName + "\" already present in file \"" + fileName +
                              "\". If you want to delete the original object and write a new RNTuple, please set "
                              "the 'fOverwriteIfExists' option to true in RSnapshotOptions.";
      throw std::invalid_argument(msg);
   }
}

void SetBranchesHelper(TTree *inputTree, TTree &outputTree, ROOT::Internal::RDF::RBranchSet &outputBranches,
                       int basketSize, const std::string &inputBranchName, const std::string &outputBranchName,
                       const std::type_info &valueTypeID, void *valueAddress, TBranch *&actionHelperBranchPtr,
                       void *&actionHelperBranchPtrAddress, bool isDefine)
{

   auto *inputBranch = SearchForBranch(inputTree, inputBranchName);

   // Respect the original bufsize and splitlevel arguments
   // In particular, by keeping splitlevel equal to 0 if this was the case for `inputBranch`, we avoid
   // writing garbage when unsplit objects cannot be written as split objects (e.g. in case of a polymorphic
   // TObject branch, see https://bit.ly/2EjLMId ).
   // A user-provided basket size value takes precedence.
   const auto bufSize = (basketSize > 0) ? basketSize : (inputBranch ? inputBranch->GetBasketSize() : 32000);
   const auto splitLevel = inputBranch ? inputBranch->GetSplitLevel() : 99;

   if (auto *outputBranch = outputBranches.Get(outputBranchName); outputBranch && valueAddress) {
      // The output branch was already created, we just need to (re)set its address
      SetBranchAddress(inputBranch, *outputBranch, actionHelperBranchPtrAddress,
                       outputBranches.IsCArray(outputBranchName), valueAddress);
      return;
   }

   auto *dictionary = TDictionary::GetDictionary(valueTypeID);
   if (dynamic_cast<TDataType *>(dictionary)) {
      // Branch of fundamental type
      CreateFundamentalTypeBranch(outputTree, outputBranchName, valueAddress, valueTypeID, outputBranches, bufSize);
      return;
   }

   if (!isDefine) {
      // Cases where we need a leaflist (e.g. C-style arrays)
      // We only enter this code path if the input value does not come from a Define/Redefine. In those cases, it is
      // not allowed to create a column of C-style array type, so that can't happen when writing the TTree. This is
      // currently what prevents writing the wrong branch output type in a scenario where the input branch of the TTree
      // is a C-style array and then the user is Redefining it with some other type (e.g. a ROOT::RVec).
      CreateCStyleArrayBranch(outputTree, outputBranches, inputBranch, outputBranchName, bufSize, valueAddress);
   }
   if (auto *arrayBranch = outputBranches.Get(outputBranchName)) {
      // A branch was created in the previous function call
      actionHelperBranchPtr = arrayBranch;
      if (valueAddress) {
         // valueAddress here points to a ROOT::RVec<std::byte> coming from RTreeUntypedArrayColumnReader. We know we
         // need its buffer, so we cast it and extract the address of the buffer
         auto *rawRVec = reinterpret_cast<ROOT::RVec<std::byte> *>(valueAddress);
         actionHelperBranchPtrAddress = rawRVec->data();
      }
      return;
   }

   if (auto *classPtr = dynamic_cast<TClass *>(dictionary)) {
      TBranch *outputBranch{};
      // Case of unsplit object with polymorphic type
      if (inputBranch && dynamic_cast<TBranchObject *>(inputBranch) && valueAddress)
         outputBranch = ROOT::Internal::TreeUtils::CallBranchImp(outputTree, outputBranchName.c_str(), classPtr,
                                                                 inputBranch->GetAddress(), bufSize, splitLevel);
      // General case, with valid address
      else if (valueAddress)
         outputBranch = ROOT::Internal::TreeUtils::CallBranchImpRef(outputTree, outputBranchName.c_str(), classPtr,
                                                                    TDataType::GetType(valueTypeID), valueAddress,
                                                                    bufSize, splitLevel);
      // No value was passed, we're just creating a hollow branch to populate the dataset schema
      else
         outputBranch = outputTree.Branch(outputBranchName.c_str(), classPtr->GetName(), nullptr, bufSize);
      outputBranches.Insert(outputBranchName, outputBranch);
      return;
   }

   // We are not aware of other cases
   throw std::logic_error(
      "RDataFrame::Snapshot: something went wrong when creating a TTree branch, please report this as a bug.");
}
} // namespace

TBranch *ROOT::Internal::RDF::RBranchSet::Get(const std::string &name) const
{
   auto it = std::find(fNames.begin(), fNames.end(), name);
   if (it == fNames.end())
      return nullptr;
   return fBranches[std::distance(fNames.begin(), it)];
}

bool ROOT::Internal::RDF::RBranchSet::IsCArray(const std::string &name) const
{
   if (auto it = std::find(fNames.begin(), fNames.end(), name); it != fNames.end())
      return fIsCArray[std::distance(fNames.begin(), it)];
   return false;
}

void ROOT::Internal::RDF::RBranchSet::Insert(const std::string &name, TBranch *address, bool isCArray)
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

void ROOT::Internal::RDF::RBranchSet::Clear()
{
   fBranches.clear();
   fNames.clear();
   fIsCArray.clear();
}

void ROOT::Internal::RDF::RBranchSet::AssertNoNullBranchAddresses()
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
      msg += " are needed as they provide the size of other branches containing dynamically sized arrays, but they are";
   }
   msg += " not part of the set of branches that are being written out.";
   throw std::runtime_error(msg);
}

ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::UntypedSnapshotTTreeHelper(
   std::string_view filename, std::string_view dirname, std::string_view treename, const ColumnNames_t &vbnames,
   const ColumnNames_t &bnames, const RSnapshotOptions &options, std::vector<bool> &&isDefine,
   ROOT::Detail::RDF::RLoopManager *loopManager, ROOT::Detail::RDF::RLoopManager *inputLM,
   const std::vector<const std::type_info *> &colTypeIDs)
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
     fInputLoopManager(inputLM),
     fInputColumnTypeIDs(colTypeIDs)
{
   EnsureValidSnapshotTTreeOutput(fOptions, fTreeName, fFileName);
}

// Define special member methods here where the definition of all the data member types is available
ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::UntypedSnapshotTTreeHelper(
   ROOT::Internal::RDF::UntypedSnapshotTTreeHelper &&) noexcept = default;
ROOT::Internal::RDF::UntypedSnapshotTTreeHelper &ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::operator=(
   ROOT::Internal::RDF::UntypedSnapshotTTreeHelper &&) noexcept = default;

ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::~UntypedSnapshotTTreeHelper()
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

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::InitTask(TTreeReader *, unsigned int)
{
   // We ask the input RLoopManager if it has a TTree. We cannot rely on getting this information when constructing
   // this action helper, since the TTree might change e.g. when ChangeSpec is called in-between distributed tasks.
   fInputTree = fInputLoopManager->GetTree();
   fBranchAddressesNeedReset = true;
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::Exec(unsigned int, const std::vector<void *> &values)
{
   if (!fBranchAddressesNeedReset) {
      UpdateCArraysPtrs(values);
   } else {
      SetBranches(values);
      fBranchAddressesNeedReset = false;
   }

   fOutputTree->Fill();
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::UpdateCArraysPtrs(const std::vector<void *> &values)
{
   // This code deals with branches which hold C arrays of variable size. It can happen that the buffers
   // associated to those is re-allocated. As a result the value of the pointer can change therewith
   // leaving associated to the branch of the output tree an invalid pointer.
   // With this code, we set the value of the pointer in the output branch anew when needed.
   assert(values.size() == fBranches.size());
   auto nValues = values.size();
   for (decltype(nValues) i{}; i < nValues; i++) {
      if (fBranches[i] && fOutputBranches.IsCArray(fOutputBranchNames[i])) {
         // valueAddress here points to a ROOT::RVec<std::byte> coming from RTreeUntypedArrayColumnReader. We know we
         // need its buffer, so we cast it and extract the address of the buffer
         auto *rawRVec = reinterpret_cast<ROOT::RVec<std::byte> *>(values[i]);
         if (auto *data = rawRVec->data(); fBranchAddresses[i] != data) {
            // reset the branch address
            fBranches[i]->SetAddress(data);
            fBranchAddresses[i] = data;
         }
      }
   }
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::SetBranches(const std::vector<void *> &values)
{
   // create branches in output tree
   auto nValues = values.size();
   for (decltype(nValues) i{}; i < nValues; i++) {
      SetBranchesHelper(fInputTree, *fOutputTree, fOutputBranches, fOptions.fBasketSize, fInputBranchNames[i],
                        fOutputBranchNames[i], *fInputColumnTypeIDs[i], values[i], fBranches[i], fBranchAddresses[i],
                        fIsDefine[i]);
   }
   fOutputBranches.AssertNoNullBranchAddresses();
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::SetEmptyBranches(TTree *inputTree, TTree &outputTree)
{
   void *dummyValueAddress{};
   TBranch *dummyTBranchPtr{};
   void *dummyTBranchAddress{};
   RBranchSet outputBranches{};
   auto nBranches = fInputBranchNames.size();
   for (decltype(nBranches) i{}; i < nBranches; i++) {
      SetBranchesHelper(inputTree, outputTree, outputBranches, fOptions.fBasketSize, fInputBranchNames[i],
                        fOutputBranchNames[i], *fInputColumnTypeIDs[i], dummyValueAddress, dummyTBranchPtr,
                        dummyTBranchAddress, fIsDefine[i]);
   }
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::Initialize()
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

   fOutputTree = std::make_unique<TTree>(fTreeName.c_str(), fTreeName.c_str(), fOptions.fSplitLevel, /*dir=*/outputDir);

   if (fOptions.fAutoFlush)
      fOutputTree->SetAutoFlush(fOptions.fAutoFlush);
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::Finalize()
{
   assert(fOutputTree != nullptr);
   assert(fOutputFile != nullptr);

   // There were no entries to fill the TTree with (either the input TTree was empty or no event passed after
   // filtering). We have already created an empty TTree, now also create the branches to preserve the schema
   if (fOutputTree->GetEntries() == 0) {
      SetEmptyBranches(fInputTree, *fOutputTree);
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

/**
 * \brief Create a new UntypedSnapshotTTreeHelper with a different output file name
 *
 * \param newName A type-erased string with the output file name
 * \return UntypedSnapshotTTreeHelper
 *
 * This MakeNew implementation is tied to the cloning feature of actions
 * of the computation graph. In particular, cloning a Snapshot node usually
 * also involves changing the name of the output file, otherwise the cloned
 * Snapshot would overwrite the same file.
 */
ROOT::Internal::RDF::UntypedSnapshotTTreeHelper
ROOT::Internal::RDF::UntypedSnapshotTTreeHelper::MakeNew(void *newName, std::string_view)
{
   const std::string finalName = *reinterpret_cast<const std::string *>(newName);
   return ROOT::Internal::RDF::UntypedSnapshotTTreeHelper{finalName,
                                                          fDirName,
                                                          fTreeName,
                                                          fInputBranchNames,
                                                          fOutputBranchNames,
                                                          fOptions,
                                                          std::vector<bool>(fIsDefine),
                                                          fOutputLoopManager,
                                                          fInputLoopManager,
                                                          fInputColumnTypeIDs};
}

ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::UntypedSnapshotTTreeHelperMT(
   unsigned int nSlots, std::string_view filename, std::string_view dirname, std::string_view treename,
   const ColumnNames_t &vbnames, const ColumnNames_t &bnames, const RSnapshotOptions &options,
   std::vector<bool> &&isDefine, ROOT::Detail::RDF::RLoopManager *loopManager, ROOT::Detail::RDF::RLoopManager *inputLM,
   const std::vector<const std::type_info *> &colTypeIDs)
   : fNSlots(nSlots),
     fOutputFiles(fNSlots),
     fOutputTrees(fNSlots),
     fBranchAddressesNeedReset(fNSlots, 1),
     fInputTrees(fNSlots),
     fBranches(fNSlots, std::vector<TBranch *>(vbnames.size(), nullptr)),
     fBranchAddresses(fNSlots, std::vector<void *>(vbnames.size(), nullptr)),
     fOutputBranches(fNSlots),
     fFileName(filename),
     fDirName(dirname),
     fTreeName(treename),
     fOptions(options),
     fOutputBranchNames(ReplaceDotWithUnderscore(bnames)),
     fOutputLoopManager(loopManager),
     fInputLoopManager(inputLM),
     fInputBranchNames(vbnames),
     fInputColumnTypeIDs(colTypeIDs),
     fIsDefine(std::move(isDefine))
{
   EnsureValidSnapshotTTreeOutput(fOptions, fTreeName, fFileName);
}

// Define special member methods here where the definition of all the data member types is available
ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::UntypedSnapshotTTreeHelperMT(
   ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT &&) noexcept = default;
ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT &ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::operator=(
   ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT &&) noexcept = default;

ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::~UntypedSnapshotTTreeHelperMT()
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

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::InitTask(TTreeReader *r, unsigned int slot)
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

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::FinalizeTask(unsigned int slot)
{
   if (fOutputTrees[slot]->GetEntries() > 0)
      fOutputFiles[slot]->Write();
   // clear now to avoid concurrent destruction of output trees and input tree (which has them listed as fClones)
   fOutputTrees[slot].reset(nullptr);
   fOutputBranches[slot].Clear();
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::Exec(unsigned int slot, const std::vector<void *> &values)
{
   if (fBranchAddressesNeedReset[slot] == 0) {
      UpdateCArraysPtrs(slot, values);
   } else {
      SetBranches(slot, values);
      fBranchAddressesNeedReset[slot] = 0;
   }
   fOutputTrees[slot]->Fill();
   auto entries = fOutputTrees[slot]->GetEntries();
   auto autoFlush = fOutputTrees[slot]->GetAutoFlush();
   if ((autoFlush > 0) && (entries % autoFlush == 0))
      fOutputFiles[slot]->Write();
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::UpdateCArraysPtrs(unsigned int slot,
                                                                          const std::vector<void *> &values)
{
   // This code deals with branches which hold C arrays of variable size. It can happen that the buffers
   // associated to those is re-allocated. As a result the value of the pointer can change therewith
   // leaving associated to the branch of the output tree an invalid pointer.
   // With this code, we set the value of the pointer in the output branch anew when needed.
   assert(values.size() == fBranches[slot].size());
   auto nValues = values.size();
   for (decltype(nValues) i{}; i < nValues; i++) {
      if (fBranches[slot][i] && fOutputBranches[slot].IsCArray(fOutputBranchNames[i])) {
         // valueAddress here points to a ROOT::RVec<std::byte> coming from RTreeUntypedArrayColumnReader. We know we
         // need its buffer, so we cast it and extract the address of the buffer
         auto *rawRVec = reinterpret_cast<ROOT::RVec<std::byte> *>(values[i]);
         if (auto *data = rawRVec->data(); fBranchAddresses[slot][i] != data) {
            // reset the branch address
            fBranches[slot][i]->SetAddress(data);
            fBranchAddresses[slot][i] = data;
         }
      }
   }
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::SetBranches(unsigned int slot,
                                                                    const std::vector<void *> &values)
{
   // create branches in output tree
   auto nValues = values.size();
   for (decltype(nValues) i{}; i < nValues; i++) {
      SetBranchesHelper(fInputTrees[slot], *fOutputTrees[slot], fOutputBranches[slot], fOptions.fBasketSize,
                        fInputBranchNames[i], fOutputBranchNames[i], *fInputColumnTypeIDs[i], values[i],
                        fBranches[slot][i], fBranchAddresses[slot][i], fIsDefine[i]);
   }
   fOutputBranches[slot].AssertNoNullBranchAddresses();
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::SetEmptyBranches(TTree *inputTree, TTree &outputTree)
{
   void *dummyValueAddress{};
   TBranch *dummyTBranchPtr{};
   void *dummyTBranchAddress{};
   RBranchSet outputBranches{};
   auto nBranches = fInputBranchNames.size();
   for (decltype(nBranches) i{}; i < nBranches; i++) {
      SetBranchesHelper(inputTree, outputTree, outputBranches, fOptions.fBasketSize, fInputBranchNames[i],
                        fOutputBranchNames[i], *fInputColumnTypeIDs[i], dummyValueAddress, dummyTBranchPtr,
                        dummyTBranchAddress, fIsDefine[i]);
   }
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::Initialize()
{
   const auto cs = ROOT::CompressionSettings(fOptions.fCompressionAlgorithm, fOptions.fCompressionLevel);
   auto outFile =
      std::unique_ptr<TFile>{TFile::Open(fFileName.c_str(), fOptions.fMode.c_str(), /*ftitle=*/fFileName.c_str(), cs)};
   if (!outFile)
      throw std::runtime_error("Snapshot: could not create output file " + fFileName);
   fOutputFile = outFile.get();
   fMerger = std::make_unique<ROOT::TBufferMerger>(std::move(outFile));
}

void ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::Finalize()
{

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
      SetEmptyBranches(fInputLoopManager->GetTree(), *outTree);

      fOutputFile->Write();
   }

   // flush all buffers to disk by destroying the TBufferMerger
   fOutputFiles.clear();
   fMerger.reset();

   // Now connect the data source to the loop manager so it can be used for further processing
   fOutputLoopManager->SetDataSource(std::make_unique<ROOT::Internal::RDF::RTTreeDS>(fullTreeName, fFileName));
}

/**
 * \brief Create a new UntypedSnapshotTTreeHelperMT with a different output file name
 *
 * \param newName A type-erased string with the output file name
 * \return UntypedSnapshotTTreeHelperMT
 *
 * This MakeNew implementation is tied to the cloning feature of actions
 * of the computation graph. In particular, cloning a Snapshot node usually
 * also involves changing the name of the output file, otherwise the cloned
 * Snapshot would overwrite the same file.
 */
ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT
ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT::MakeNew(void *newName, std::string_view)
{
   const std::string finalName = *reinterpret_cast<const std::string *>(newName);
   return ROOT::Internal::RDF::UntypedSnapshotTTreeHelperMT{fNSlots,
                                                            finalName,
                                                            fDirName,
                                                            fTreeName,
                                                            fInputBranchNames,
                                                            fOutputBranchNames,
                                                            fOptions,
                                                            std::vector<bool>(fIsDefine),
                                                            fOutputLoopManager,
                                                            fInputLoopManager,
                                                            fInputColumnTypeIDs};
}

ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper::UntypedSnapshotRNTupleHelper(
   std::string_view filename, std::string_view dirname, std::string_view ntuplename, const ColumnNames_t &vfnames,
   const ColumnNames_t &fnames, const RSnapshotOptions &options, ROOT::Detail::RDF::RLoopManager *inputLM,
   ROOT::Detail::RDF::RLoopManager *outputLM, std::vector<bool> &&isDefine,
   const std::vector<const std::type_info *> &colTypeIDs)
   : fFileName(filename),
     fDirName(dirname),
     fNTupleName(ntuplename),
     fOutputFile(nullptr),
     fOptions(options),
     fInputLoopManager(inputLM),
     fOutputLoopManager(outputLM),
     fInputFieldNames(vfnames),
     fOutputFieldNames(ReplaceDotWithUnderscore(fnames)),
     fWriter(nullptr),
     fOutputEntry(nullptr),
     fIsDefine(std::move(isDefine)),
     fInputColumnTypeIDs(colTypeIDs)
{
   EnsureValidSnapshotRNTupleOutput(fOptions, fNTupleName, fFileName);
}

// Define special member methods here where the definition of all the data member types is available
ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper::UntypedSnapshotRNTupleHelper(
   ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper &&) noexcept = default;
ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper &ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper::operator=(
   ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper &&) noexcept = default;

ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper::~UntypedSnapshotRNTupleHelper()
{
   if (!fNTupleName.empty() && !fOutputLoopManager->GetDataSource() && fOptions.fLazy)
      Warning("Snapshot", "A lazy Snapshot action was booked but never triggered.");
}

void ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper::Exec(unsigned int /* slot */, const std::vector<void *> &values)
{
   assert(values.size() == fOutputFieldNames.size());
   for (decltype(values.size()) i = 0; i < values.size(); i++) {
      fOutputEntry->BindRawPtr(fOutputFieldNames[i], values[i]);
   }
   fWriter->Fill();
}

void ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper::Initialize()
{
   auto model = ROOT::RNTupleModel::Create();
   auto nFields = fOutputFieldNames.size();
   for (decltype(nFields) i = 0; i < nFields; i++) {
      // Need to retrieve the type of every field to create as a string
      // If the input type for a field does not have RTTI, internally we store it as the tag UseNativeDataType. When
      // that is detected, we need to ask the data source which is the type name based on the on-disk information.
      const auto typeName = *fInputColumnTypeIDs[i] == typeid(ROOT::Internal::RDF::UseNativeDataType)
                               ? ROOT::Internal::RDF::GetTypeNameWithOpts(*fInputLoopManager->GetDataSource(),
                                                                          fInputFieldNames[i], fOptions.fVector2RVec)
                               : ROOT::Internal::RDF::TypeID2TypeName(*fInputColumnTypeIDs[i]);
      model->AddField(ROOT::RFieldBase::Create(fOutputFieldNames[i], typeName).Unwrap());
   }
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

void ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper::Finalize()
{
   fWriter.reset();
   // We can now set the data source of the loop manager for the RDataFrame that is returned by the Snapshot call.
   fOutputLoopManager->SetDataSource(std::make_unique<ROOT::RDF::RNTupleDS>(fDirName + "/" + fNTupleName, fFileName));
}

/**
 * Create a new UntypedSnapshotRNTupleHelper with a different output file name.
 *
 * \param[in] newName A type-erased string with the output file name
 * \return UntypedSnapshotRNTupleHelper
 *
 * This MakeNew implementation is tied to the cloning feature of actions
 * of the computation graph. In particular, cloning a Snapshot node usually
 * also involves changing the name of the output file, otherwise the cloned
 * Snapshot would overwrite the same file.
 */
ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper
ROOT::Internal::RDF::UntypedSnapshotRNTupleHelper::MakeNew(void *newName)
{
   const std::string finalName = *reinterpret_cast<const std::string *>(newName);
   return UntypedSnapshotRNTupleHelper{finalName,          fDirName,           fNTupleName,
                                       fInputFieldNames,   fOutputFieldNames,  fOptions,
                                       fInputLoopManager,  fOutputLoopManager, std::vector<bool>(fIsDefine),
                                       fInputColumnTypeIDs};
}
