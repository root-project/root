#include "BranchCountStruct.h"

#include "TBranchElement.h"
#include "TClonesArray.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TSystem.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

struct FileGuard {
   std::string fName;
   explicit FileGuard(std::string name) : fName(std::move(name)) {}
   ~FileGuard() { gSystem->Unlink(fName.c_str()); }
};

// Writes one entry with one branch per (name, size) pair. The size is passed to T::Set.
template <typename T>
void WriteObjects(const char *fileName, const std::vector<std::pair<std::string, int>> &branches)
{
   TFile file(fileName, "RECREATE");
   TTree tree("t", "t");
   std::vector<std::unique_ptr<T>> objects;
   std::vector<T *> addresses(branches.size());
   for (std::size_t i = 0; i < branches.size(); ++i) {
      objects.emplace_back(std::make_unique<T>());
      addresses[i] = objects.back().get();
      tree.Branch(branches[i].first.c_str(), &addresses[i], 32000, 99);
      objects.back()->Set(branches[i].second);
   }
   tree.Fill();
   tree.Write();
   tree.ResetBranchAddresses();
}

// The counted-array branch `arrayName` (full name, e.g. "b.fX") of top-level branch `top` must take
// its counter from a sub-branch of the same top-level branch, both on the branch and on its leaf.
void ExpectOwnCounter(TTree &tree, const char *top, const char *arrayName)
{
   TBranch *topBranch = tree.GetBranch(top);
   ASSERT_NE(topBranch, nullptr);
   auto *array = dynamic_cast<TBranchElement *>(tree.GetBranch(arrayName));
   ASSERT_NE(array, nullptr);
   ASSERT_EQ(array->GetMother(), topBranch);

   TBranchElement *counter = array->GetBranchCount();
   ASSERT_NE(counter, nullptr);
   EXPECT_EQ(counter->GetMother(), topBranch) << "counter is under " << counter->GetMother()->GetName();

   auto *leaf = static_cast<TLeaf *>(array->GetListOfLeaves()->At(0));
   ASSERT_NE(leaf, nullptr);
   ASSERT_NE(leaf->GetLeafCount(), nullptr);
   EXPECT_EQ(leaf->GetLeafCount()->GetBranch()->GetMother(), topBranch);
}

// Counted arrays inside a TClonesArray or STL collection count elements with the collection
// branch and take their per-element size from the prefixed `<top>.fN` sub-branch.
void ExpectCollectionCounters(TTree &tree, const std::string &top)
{
   TBranch *topBranch = tree.GetBranch(top.c_str());
   ASSERT_NE(topBranch, nullptr);
   auto *array = dynamic_cast<TBranchElement *>(tree.GetBranch((top + ".fX").c_str()));
   ASSERT_NE(array, nullptr);
   EXPECT_EQ(array->GetBranchCount(), topBranch) << top;
   ASSERT_NE(array->GetBranchCount2(), nullptr) << top;
   EXPECT_EQ(std::string(array->GetBranchCount2()->GetName()), top + ".fN");
   EXPECT_EQ(array->GetBranchCount2()->GetMother(), topBranch) << top;
}

void ExpectDrawRows(TTree &tree, const char *expression, Long64_t rows)
{
   tree.Draw(expression, "", "goff");
   EXPECT_EQ(tree.GetSelectedRows(), rows) << expression;
}

void ExpectReaderValues(TTree &tree, const char *expression)
{
   TTreeReader reader(&tree);
   TTreeReaderArray<double> values(reader, expression);
   ASSERT_TRUE(reader.Next());
   ASSERT_EQ(values.GetSize(), 3u) << expression;
   for (std::size_t i = 0; i < 3; ++i)
      EXPECT_DOUBLE_EQ(values[i], i);
}

} // namespace

// a holds 0 elements and b holds 3, so reading b's array with a's counter gives 0 elements.

TEST(TBranchElementBranchCount, UnprefixedBasicPointer)
{
   FileGuard guard("branchcount_unprefixed_basic.root");
   WriteObjects<BCVertex>(guard.fName.c_str(), {{"a", 0}, {"b", 3}});

   TFile file(guard.fName.c_str());
   auto tree = file.Get<TTree>("t");
   ASSERT_NE(tree, nullptr);
   ExpectOwnCounter(*tree, "b", "b.fX");
   ExpectDrawRows(*tree, "b.fX", 3);
}

TEST(TBranchElementBranchCount, UnprefixedObjectArray)
{
   FileGuard guard("branchcount_unprefixed_loop.root");
   WriteObjects<BCHitList>(guard.fName.c_str(), {{"a", 0}, {"b", 3}});

   TFile file(guard.fName.c_str());
   auto tree = file.Get<TTree>("t");
   ASSERT_NE(tree, nullptr);
   ExpectOwnCounter(*tree, "b", "b.fH");
}

TEST(TBranchElementBranchCount, UnprefixedEmbeddedMember)
{
   FileGuard guard("branchcount_unprefixed_embedded.root");
   WriteObjects<BCEvent>(guard.fName.c_str(), {{"a", 0}, {"b", 3}});

   TFile file(guard.fName.c_str());
   auto tree = file.Get<TTree>("t");
   ASSERT_NE(tree, nullptr);
   ExpectOwnCounter(*tree, "b", "b.fV.fX");
   ExpectDrawRows(*tree, "b.fV.fX", 3);
}

// Controls: layouts where the counter was already found correctly.

TEST(TBranchElementBranchCount, PrefixedBasicPointer)
{
   FileGuard guard("branchcount_prefixed_basic.root");
   WriteObjects<BCVertex>(guard.fName.c_str(), {{"a.", 0}, {"b.", 3}});

   TFile file(guard.fName.c_str());
   auto tree = file.Get<TTree>("t");
   ASSERT_NE(tree, nullptr);
   ExpectOwnCounter(*tree, "b.", "b.fX");
   ExpectReaderValues(*tree, "b.fX");
   ExpectDrawRows(*tree, "b.fX", 3);
}

TEST(TBranchElementBranchCount, SingleObject)
{
   FileGuard guard("branchcount_single.root");
   WriteObjects<BCVertex>(guard.fName.c_str(), {{"b", 3}});

   TFile file(guard.fName.c_str());
   auto tree = file.Get<TTree>("t");
   ASSERT_NE(tree, nullptr);
   ExpectOwnCounter(*tree, "b", "b.fX");
   ExpectReaderValues(*tree, "b.fX");
}

TEST(TBranchElementBranchCount, CollectionsUnchanged)
{
   FileGuard guard("branchcount_collections.root");
   {
      TFile file(guard.fName.c_str(), "RECREATE");
      TTree tree("t", "t");
      auto clonesA = std::make_unique<TClonesArray>("BCVertex");
      auto clonesB = std::make_unique<TClonesArray>("BCVertex");
      auto vectorA = std::make_unique<std::vector<BCVertex>>(1);
      auto vectorB = std::make_unique<std::vector<BCVertex>>(1);
      auto *pClonesA = clonesA.get();
      auto *pClonesB = clonesB.get();
      auto *pVectorA = vectorA.get();
      auto *pVectorB = vectorB.get();
      tree.Branch("ca", &pClonesA, 32000, 99);
      tree.Branch("cb", &pClonesB, 32000, 99);
      tree.Branch("va", &pVectorA, 32000, 99);
      tree.Branch("vb", &pVectorB, 32000, 99);
      static_cast<BCVertex *>(clonesA->ConstructedAt(0))->Set(0);
      static_cast<BCVertex *>(clonesB->ConstructedAt(0))->Set(3);
      (*vectorA)[0].Set(0);
      (*vectorB)[0].Set(3);
      tree.Fill();
      tree.Write();
      tree.ResetBranchAddresses();
   }

   TFile file(guard.fName.c_str());
   auto tree = file.Get<TTree>("t");
   ASSERT_NE(tree, nullptr);
   for (const char *top : {"ca", "cb", "va", "vb"})
      ExpectCollectionCounters(*tree, top);
   ExpectDrawRows(*tree, "cb.fX", 3);
}
