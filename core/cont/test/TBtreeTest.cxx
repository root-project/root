#include "gtest/gtest.h"
#include "TBtree.h"
#include "TNamed.h"
#include "TObjString.h"

#include <ROOT/TestSupport.hxx>

#include <string>
#include <vector>

TEST(TBtreeTest, DefaultConstructor)
{
   TBtree tree;
   EXPECT_EQ(0, tree.GetSize());
   EXPECT_TRUE(tree.IsEmpty());
   EXPECT_FALSE(tree.IsOwner());
   EXPECT_EQ(3, tree.Order());
}

TEST(TBtreeTest, CustomOrder)
{
   TBtree tree(5);
   EXPECT_EQ(0, tree.GetSize());
   EXPECT_TRUE(tree.IsEmpty());
   EXPECT_EQ(5, tree.Order());

   // Minimum order is 3; initializing with order < 3 issues a warning and clamps to 3
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kWarning, "Init", "order must be at least 3", /*matchFullMessage=*/false);
      TBtree treeSmall(2);
      EXPECT_EQ(3, treeSmall.Order());
   }
}

TEST(TBtreeTest, SortedInsertionAndLookup)
{
   TBtree tree;
   TNamed n3("gamma", "3");
   TNamed n1("alpha", "1");
   TNamed n4("delta", "4");
   TNamed n2("beta", "2");

   tree.Add(&n3);
   tree.Add(&n1);
   tree.Add(&n4);
   tree.Add(&n2);

   EXPECT_EQ(4, tree.GetSize());
   EXPECT_FALSE(tree.IsEmpty());

   // Elements should be maintained in sorted order
   EXPECT_EQ(&n1, tree.First());
   EXPECT_EQ(&n1, tree.At(0));
   EXPECT_EQ(&n1, tree[0]);

   EXPECT_EQ(&n2, tree.At(1));
   EXPECT_EQ(&n2, tree[1]);

   EXPECT_EQ(&n4, tree.At(2));
   EXPECT_EQ(&n4, tree[2]);

   EXPECT_EQ(&n3, tree.Last());
   EXPECT_EQ(&n3, tree.At(3));
   EXPECT_EQ(&n3, tree[3]);

   // Lookup by object pointer
   EXPECT_EQ(&n2, tree.FindObject(&n2));

   // Lookup by name
   EXPECT_EQ(&n4, tree.FindObject("delta"));

   // Lookup non-existent
   TNamed notFound("epsilon", "5");
   EXPECT_EQ(nullptr, tree.FindObject(&notFound));
   EXPECT_EQ(nullptr, tree.FindObject("not_in_tree"));
}

TEST(TBtreeTest, Rank)
{
   TBtree tree;
   TNamed n3("gamma", "3");
   TNamed n1("alpha", "1");
   TNamed n2("beta", "2");

   tree.Add(&n3);
   tree.Add(&n1);
   tree.Add(&n2);

   EXPECT_EQ(3, tree.GetSize());
   EXPECT_EQ(0, tree.Rank(&n1));
   EXPECT_EQ(1, tree.Rank(&n2));
   EXPECT_EQ(2, tree.Rank(&n3));

   TNamed notFound("omega", "99");
   EXPECT_EQ(-1, tree.Rank(&notFound));
}

TEST(TBtreeTest, NodeSplittingAndLargeInsertion)
{
   // Order 3 means leaf max index is 2*(3+1)-1 = 7.
   // Inserting 50 elements forces multiple leaf and inner node splits.
   TBtree tree(3);
   const int n = 50;
   std::vector<TNamed> items;
   items.reserve(n);

   // Insert in reverse order to exercise balancing/splitting
   for (int i = n - 1; i >= 0; --i) {
      char key[16];
      std::snprintf(key, sizeof(key), "key_%03d", i);
      items.emplace_back(key, "val");
   }

   for (auto &item : items) {
      tree.Add(&item);
   }

   EXPECT_EQ(n, tree.GetSize());

   // Verify elements are strictly sorted
   for (int i = 0; i < n; ++i) {
      char expectedKey[16];
      std::snprintf(expectedKey, sizeof(expectedKey), "key_%03d", i);
      TObject *obj = tree.At(i);
      ASSERT_NE(nullptr, obj);
      EXPECT_STREQ(expectedKey, obj->GetName());
      EXPECT_EQ(i, tree.Rank(obj));
   }

   // Remove every second element
   for (int i = 0; i < n; i += 2) {
      char expectedKey[16];
      std::snprintf(expectedKey, sizeof(expectedKey), "key_%03d", i);
      TNamed target(expectedKey, "val");
      TObject *removed = tree.Remove(&target);
      EXPECT_NE(nullptr, removed);
      EXPECT_STREQ(expectedKey, removed->GetName());
   }

   EXPECT_EQ(n / 2, tree.GetSize());

   // Verify remaining elements are still strictly sorted
   for (int i = 0; i < n / 2; ++i) {
      int expectedIndex = 2 * i + 1;
      char expectedKey[16];
      std::snprintf(expectedKey, sizeof(expectedKey), "key_%03d", expectedIndex);
      TObject *obj = tree.At(i);
      ASSERT_NE(nullptr, obj);
      EXPECT_STREQ(expectedKey, obj->GetName());
   }
}

TEST(TBtreeTest, RemoveAndClear)
{
   TBtree tree;
   TNamed a("A", "1");
   TNamed b("B", "2");
   TNamed c("C", "3");

   tree.Add(&a);
   tree.Add(&b);
   tree.Add(&c);
   EXPECT_EQ(3, tree.GetSize());

   // Remove middle element
   EXPECT_EQ(&b, tree.Remove(&b));
   EXPECT_EQ(2, tree.GetSize());
   EXPECT_EQ(nullptr, tree.FindObject(&b));
   EXPECT_EQ(&a, tree.At(0));
   EXPECT_EQ(&c, tree.At(1));

   // Remove non-existent object
   TNamed notInTree("X", "9");
   EXPECT_EQ(nullptr, tree.Remove(&notInTree));
   EXPECT_EQ(2, tree.GetSize());

   // Clear leaves objects intact
   tree.Clear();
   EXPECT_EQ(0, tree.GetSize());
   EXPECT_TRUE(tree.IsEmpty());
}

namespace {
class TrackedObj : public TNamed {
public:
   static int sAliveCount;
   TrackedObj(const char *name) : TNamed(name, "") { ++sAliveCount; }
   ~TrackedObj() override { --sAliveCount; }
};
int TrackedObj::sAliveCount = 0;
} // namespace

TEST(TBtreeTest, OwnershipAndClearDelete)
{
   TrackedObj::sAliveCount = 0;

   // Non-owning tree: Clear does not destroy objects
   {
      TBtree tree;
      auto *obj1 = new TrackedObj("obj1");
      auto *obj2 = new TrackedObj("obj2");
      tree.Add(obj1);
      tree.Add(obj2);
      EXPECT_EQ(2, TrackedObj::sAliveCount);

      tree.Clear();
      EXPECT_EQ(0, tree.GetSize());
      EXPECT_EQ(2, TrackedObj::sAliveCount);

      tree.Add(obj1);
      tree.Add(obj2);
      tree.Delete();
      EXPECT_EQ(0, tree.GetSize());
      EXPECT_EQ(0, TrackedObj::sAliveCount);
   }

   // Owning tree: Clear destroys owned objects
   TrackedObj::sAliveCount = 0;
   {
      TBtree tree;
      tree.SetOwner(kTRUE);
      EXPECT_TRUE(tree.IsOwner());

      tree.Add(new TrackedObj("obj1"));
      tree.Add(new TrackedObj("obj2"));
      EXPECT_EQ(2, TrackedObj::sAliveCount);

      tree.Clear();
      EXPECT_EQ(0, tree.GetSize());
      EXPECT_EQ(0, TrackedObj::sAliveCount);

      tree.Add(new TrackedObj("obj3"));
      tree.Add(new TrackedObj("obj4"));
      EXPECT_EQ(2, TrackedObj::sAliveCount);
   }
   // Destructor cleans up owned objects
   EXPECT_EQ(0, TrackedObj::sAliveCount);
}

TEST(TBtreeTest, Iteration)
{
   TBtree tree;
   TNamed a("A", "1");
   TNamed b("B", "2");
   TNamed c("C", "3");

   tree.Add(&c);
   tree.Add(&a);
   tree.Add(&b);

   // Forward iteration
   {
      TIter next(&tree);
      EXPECT_EQ(&a, next());
      EXPECT_EQ(&b, next());
      EXPECT_EQ(&c, next());
      EXPECT_EQ(nullptr, next());

      next.Reset();
      EXPECT_EQ(&a, next());
   }

   // Backward iteration
   {
      TIter next(&tree, kIterBackward);
      EXPECT_EQ(&c, next());
      EXPECT_EQ(&b, next());
      EXPECT_EQ(&a, next());
      EXPECT_EQ(nullptr, next());
   }

   // TBtreeIter direct usage
   {
      TBtreeIter iter(&tree);
      EXPECT_EQ(&tree, iter.GetCollection());
      EXPECT_EQ(&a, iter.Next());
      EXPECT_EQ(&a, *iter);
      EXPECT_EQ(&b, iter.Next());
      EXPECT_EQ(&b, *iter);
      EXPECT_EQ(&c, iter.Next());
      EXPECT_EQ(&c, *iter);
      EXPECT_EQ(nullptr, iter.Next());
   }
}

TEST(TBtreeTest, ErrorDiagnosticsForNonSortable)
{
   TBtree tree;
   TObject nonSortable; // TObject::IsSortable() returns kFALSE

   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "Add", "object must be sortable", /*matchFullMessage=*/false);
      tree.Add(&nonSortable);
   }

   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "FindObject", "object must be sortable", /*matchFullMessage=*/false);
      EXPECT_EQ(nullptr, tree.FindObject(&nonSortable));
   }

   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "Rank", "object must be sortable", /*matchFullMessage=*/false);
      EXPECT_EQ(-1, tree.Rank(&nonSortable));
   }

   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "Remove", "object must be sortable", /*matchFullMessage=*/false);
      EXPECT_EQ(nullptr, tree.Remove(&nonSortable));
   }

   // Before and After are forbidden because Btree controls sorting order
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "TBtree::Before", "May not use this method", /*matchFullMessage=*/false);
      EXPECT_EQ(nullptr, tree.Before(&nonSortable));
   }
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "TBtree::After", "Cannot use this method", /*matchFullMessage=*/false);
      EXPECT_EQ(nullptr, tree.After(&nonSortable));
   }
}
