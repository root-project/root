// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/ActionHelpers.hxx"
#include "ROOT/RDF/Utils.hxx" // CacheLineStep
#include "ROOT/RNTuple.hxx"   // ValidateSnapshotRNTupleOutput

namespace ROOT {
namespace Internal {
namespace RDF {

void ResetIfPossible(TStatistic *h)
{
   *h = TStatistic();
}
// cannot safely re-initialize variations of the result, hence error out
void ResetIfPossible(...)
{
   throw std::runtime_error(
      "A systematic variation was requested for a custom Fill action, but the type of the object to be filled does "
      "not implement a Reset method, so we cannot safely re-initialize variations of the result. Aborting.");
}

void UnsetDirectoryIfPossible(TH1 *h)
{
   h->SetDirectory(nullptr);
}
void UnsetDirectoryIfPossible(...) {}

CountHelper::CountHelper(const std::shared_ptr<ULong64_t> &resultCount, const unsigned int nSlots)
   : fResultCount(resultCount), fCounts(nSlots, 0)
{
}

void CountHelper::Exec(unsigned int slot)
{
   fCounts[slot]++;
}

void CountHelper::Finalize()
{
   *fResultCount = 0;
   for (auto &c : fCounts) {
      *fResultCount += c;
   }
}

ULong64_t &CountHelper::PartialUpdate(unsigned int slot)
{
   return fCounts[slot];
}

void BufferedFillHelper::UpdateMinMax(unsigned int slot, double v)
{
   auto &thisMin = fMin[slot * CacheLineStep<BufEl_t>()];
   auto &thisMax = fMax[slot * CacheLineStep<BufEl_t>()];
   thisMin = std::min(thisMin, v);
   thisMax = std::max(thisMax, v);
}

BufferedFillHelper::BufferedFillHelper(const std::shared_ptr<Hist_t> &h, const unsigned int nSlots)
   : fResultHist(h), fNSlots(nSlots), fBufSize(fgTotalBufSize / nSlots), fPartialHists(fNSlots),
     fMin(nSlots * CacheLineStep<BufEl_t>(), std::numeric_limits<BufEl_t>::max()),
     fMax(nSlots * CacheLineStep<BufEl_t>(), std::numeric_limits<BufEl_t>::lowest())
{
   fBuffers.reserve(fNSlots);
   fWBuffers.reserve(fNSlots);
   for (unsigned int i = 0; i < fNSlots; ++i) {
      Buf_t v;
      v.reserve(fBufSize);
      fBuffers.emplace_back(v);
      fWBuffers.emplace_back(v);
   }
}

void BufferedFillHelper::Exec(unsigned int slot, double v)
{
   UpdateMinMax(slot, v);
   fBuffers[slot].emplace_back(v);
}

void BufferedFillHelper::Exec(unsigned int slot, double v, double w)
{
   UpdateMinMax(slot, v);
   fBuffers[slot].emplace_back(v);
   fWBuffers[slot].emplace_back(w);
}

Hist_t &BufferedFillHelper::PartialUpdate(unsigned int slot)
{
   auto &partialHist = fPartialHists[slot];
   // TODO it is inefficient to re-create the partial histogram everytime the callback is called
   //      ideally we could incrementally fill it with the latest entries in the buffers
   partialHist = std::make_unique<Hist_t>(*fResultHist);
   auto weights = fWBuffers[slot].empty() ? nullptr : fWBuffers[slot].data();
   partialHist->FillN(fBuffers[slot].size(), fBuffers[slot].data(), weights);
   return *partialHist;
}

void BufferedFillHelper::Finalize()
{
   for (unsigned int i = 0; i < fNSlots; ++i) {
      if (!fWBuffers[i].empty() && fBuffers[i].size() != fWBuffers[i].size()) {
         throw std::runtime_error("Cannot fill weighted histogram with values in containers of different sizes.");
      }
   }

   BufEl_t globalMin = *std::min_element(fMin.begin(), fMin.end());
   BufEl_t globalMax = *std::max_element(fMax.begin(), fMax.end());

   if (fResultHist->CanExtendAllAxes() && globalMin != std::numeric_limits<BufEl_t>::max() &&
       globalMax != std::numeric_limits<BufEl_t>::lowest()) {
      fResultHist->SetBins(fResultHist->GetNbinsX(), globalMin, globalMax);
   }

   for (unsigned int i = 0; i < fNSlots; ++i) {
      auto weights = fWBuffers[i].empty() ? nullptr : fWBuffers[i].data();
      fResultHist->FillN(fBuffers[i].size(), fBuffers[i].data(), weights);
   }
}

MeanHelper::MeanHelper(const std::shared_ptr<double> &meanVPtr, const unsigned int nSlots)
   : fResultMean(meanVPtr), fCounts(nSlots, 0), fSums(nSlots, 0), fPartialMeans(nSlots), fCompensations(nSlots)
{
}

void MeanHelper::Exec(unsigned int slot, double v)
{
   fCounts[slot]++;
   // Kahan Sum:
   double y = v - fCompensations[slot];
   double t = fSums[slot] + y;
   fCompensations[slot] = (t - fSums[slot]) - y;
   fSums[slot] = t;
}

void MeanHelper::Finalize()
{
   double sumOfSums = 0;
   // Kahan Sum:
   double compensation(0);
   double y(0);
   double t(0);
   for (auto &m : fSums) {
      y = m - compensation;
      t = sumOfSums + y;
      compensation = (t - sumOfSums) - y;
      sumOfSums = t;
   }
   ULong64_t sumOfCounts = 0;
   for (auto &c : fCounts)
      sumOfCounts += c;
   *fResultMean = sumOfSums / (sumOfCounts > 0 ? sumOfCounts : 1);
}

double &MeanHelper::PartialUpdate(unsigned int slot)
{
   fPartialMeans[slot] = fSums[slot] / fCounts[slot];
   return fPartialMeans[slot];
}

StdDevHelper::StdDevHelper(const std::shared_ptr<double> &meanVPtr, const unsigned int nSlots)
   : fNSlots(nSlots), fResultStdDev(meanVPtr), fCounts(nSlots, 0), fMeans(nSlots, 0), fDistancesfromMean(nSlots, 0)
{
}

void StdDevHelper::Exec(unsigned int slot, double v)
{
   // Applies the Welford's algorithm to the stream of values received by the thread
   auto count = ++fCounts[slot];
   auto delta = v - fMeans[slot];
   auto mean = fMeans[slot] + delta / count;
   auto delta2 = v - mean;
   auto distance = fDistancesfromMean[slot] + delta * delta2;

   fCounts[slot] = count;
   fMeans[slot] = mean;
   fDistancesfromMean[slot] = distance;
}

void StdDevHelper::Finalize()
{
   // Evaluates and merges the partial result of each set of data to get the overall standard deviation.
   double totalElements = 0;
   for (auto c : fCounts) {
      totalElements += c;
   }
   if (totalElements == 0 || totalElements == 1) {
      // Std deviation is not defined for 1 element.
      *fResultStdDev = 0;
      return;
   }

   double overallMean = 0;
   for (unsigned int i = 0; i < fNSlots; ++i) {
      overallMean += fCounts[i] * fMeans[i];
   }
   overallMean = overallMean / totalElements;

   double variance = 0;
   for (unsigned int i = 0; i < fNSlots; ++i) {
      if (fCounts[i] == 0) {
         continue;
      }
      auto setVariance = fDistancesfromMean[i] / (fCounts[i]);
      variance += (fCounts[i]) * (setVariance + std::pow((fMeans[i] - overallMean), 2));
   }

   variance = variance / (totalElements - 1);
   *fResultStdDev = std::sqrt(variance);
}

// External templates are disabled for gcc5 since this version wrongly omits the C++11 ABI attribute
#if __GNUC__ > 5
template class TakeHelper<bool, bool, std::vector<bool>>;
template class TakeHelper<unsigned int, unsigned int, std::vector<unsigned int>>;
template class TakeHelper<unsigned long, unsigned long, std::vector<unsigned long>>;
template class TakeHelper<unsigned long long, unsigned long long, std::vector<unsigned long long>>;
template class TakeHelper<int, int, std::vector<int>>;
template class TakeHelper<long, long, std::vector<long>>;
template class TakeHelper<long long, long long, std::vector<long long>>;
template class TakeHelper<float, float, std::vector<float>>;
template class TakeHelper<double, double, std::vector<double>>;
#endif

void EnsureValidSnapshotTTreeOutput(const RSnapshotOptions &opts, const std::string &treeName,
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

void EnsureValidSnapshotRNTupleOutput(const RSnapshotOptions &opts, const std::string &ntupleName,
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

} // end NS RDF
} // end NS Internal
} // end NS ROOT

namespace {
void CreateCStyleArrayBranch(TTree *inputTree, TTree &outputTree, ROOT::Internal::RDF::RBranchSet &outputBranches,
                             const std::string &inputBranchName, const std::string &outputBranchName, int basketSize)
{
   TBranch *inputBranch = nullptr;
   if (inputTree) {
      inputBranch = inputTree->GetBranch(inputBranchName.c_str());
      if (!inputBranch) // try harder
         inputBranch = inputTree->FindBranch(inputBranchName.c_str());
   }
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
      auto *outputBranch =
         outputTree.Branch(outputBranchName.c_str(), static_cast<void *>(nullptr), leaflist.c_str(), bufSize);
      outputBranch->SetTitle(inputBranch->GetTitle());
      outputBranches.Insert(outputBranchName, outputBranch);
   }
}
} // namespace

void ROOT::Internal::RDF::SetEmptyBranchesHelper(TTree *inputTree, TTree &outputTree,
                                                 ROOT::Internal::RDF::RBranchSet &outputBranches,
                                                 const std::string &inputBranchName,
                                                 const std::string &outputBranchName, const std::type_info &typeInfo,
                                                 int basketSize)
{
   const auto bufSize = (basketSize > 0) ? basketSize : 32000;
   auto *classPtr = TClass::GetClass(typeInfo);
   if (!classPtr) {
      // Case of a leaflist of fundamental type, logic taken from
      // TTree::BranchImpRef(const char* branchname, TClass* ptrClass, EDataType datatype, void* addobj, Int_t bufsize,
      // Int_t splitlevel)
      auto typeName = ROOT::Internal::RDF::TypeID2TypeName(typeInfo);
      auto rootTypeChar = ROOT::Internal::RDF::TypeName2ROOTTypeName(typeName);
      if (rootTypeChar == ' ') {
         Warning(
            "Snapshot",
            "RDataFrame::Snapshot: could not correctly construct a leaflist for fundamental type in column %s. This "
            "column will not be written out.",
            outputBranchName.c_str());
         return;
      }
      std::string leafList{outputBranchName + '/' + rootTypeChar};
      auto *outputBranch =
         outputTree.Branch(outputBranchName.c_str(), static_cast<void *>(nullptr), leafList.c_str(), bufSize);
      outputBranches.Insert(outputBranchName, outputBranch);
      return;
   }

   // Find if there is an input branch, check for cases where we need a leaflist (e.g. C-style arrays)
   CreateCStyleArrayBranch(inputTree, outputTree, outputBranches, inputBranchName, outputBranchName, basketSize);

   // General case
   if (!outputBranches.Get(outputBranchName)) {
      auto *outputBranch = outputTree.Branch(outputBranchName.c_str(), classPtr->GetName(), nullptr, bufSize);
      outputBranches.Insert(outputBranchName, outputBranch);
   }
}
