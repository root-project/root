#include "gtest/gtest.h"
#include "THashList.h"
#include "TNamed.h"

namespace {
class TrackNamed : public TNamed {
public:
   static int fgCount;
   TrackNamed(const char *name, const char *title) : TNamed(name, title) { fgCount++; }
   ~TrackNamed() override { fgCount--; }
};
int TrackNamed::fgCount = 0;
} // namespace

TEST(THashListTest, DefaultConstructor)
{
   THashList list;
   EXPECT_EQ(list.GetSize(), 0);
   EXPECT_TRUE(list.IsEmpty());
   EXPECT_EQ(list.First(), nullptr);
   EXPECT_EQ(list.Last(), nullptr);
   EXPECT_EQ(list.FindObject("nonexistent"), nullptr);
}

TEST(THashListTest, AddAndFind)
{
   THashList list;
   list.SetOwner(kTRUE);

   auto *o1 = new TNamed("apple", "fruit");
   auto *o2 = new TNamed("banana", "fruit");
   auto *o3 = new TNamed("cherry", "fruit");

   list.Add(o1);
   list.Add(o2);
   list.Add(o3);

   EXPECT_EQ(list.GetSize(), 3);
   EXPECT_FALSE(list.IsEmpty());

   // Find by name
   EXPECT_EQ(list.FindObject("apple"), o1);
   EXPECT_EQ(list.FindObject("banana"), o2);
   EXPECT_EQ(list.FindObject("cherry"), o3);
   EXPECT_EQ(list.FindObject("grape"), nullptr);

   // Find by object pointer
   EXPECT_EQ(list.FindObject(o1), o1);
   EXPECT_EQ(list.FindObject(o2), o2);
   EXPECT_EQ(list.FindObject(o3), o3);

   TNamed dummy("grape", "fruit");
   EXPECT_EQ(list.FindObject(&dummy), nullptr);
}

TEST(THashListTest, OrderPreservation)
{
   THashList list;
   list.SetOwner(kTRUE);

   auto *o1 = new TNamed("first", "1");
   auto *o2 = new TNamed("second", "2");
   auto *o3 = new TNamed("third", "3");
   auto *o4 = new TNamed("fourth", "4");

   list.AddLast(o2);
   list.AddFirst(o1);
   list.AddAfter(o2, o4);
   list.AddBefore(o4, o3);

   // Order in list should be: o1, o2, o3, o4
   EXPECT_EQ(list.GetSize(), 4);
   EXPECT_EQ(list.First(), o1);
   EXPECT_EQ(list.Last(), o4);
   EXPECT_EQ(list.At(0), o1);
   EXPECT_EQ(list.At(1), o2);
   EXPECT_EQ(list.At(2), o3);
   EXPECT_EQ(list.At(3), o4);

   // All should still be findable by hash lookup
   EXPECT_EQ(list.FindObject("first"), o1);
   EXPECT_EQ(list.FindObject("second"), o2);
   EXPECT_EQ(list.FindObject("third"), o3);
   EXPECT_EQ(list.FindObject("fourth"), o4);
}

TEST(THashListTest, Remove)
{
   THashList list;
   list.SetOwner(kTRUE);

   auto *o1 = new TNamed("item1", "desc1");
   auto *o2 = new TNamed("item2", "desc2");
   auto *o3 = new TNamed("item3", "desc3");

   list.Add(o1);
   list.Add(o2);
   list.Add(o3);

   // Remove middle element
   TObject *removed = list.Remove(o2);
   EXPECT_EQ(removed, o2);
   EXPECT_EQ(list.GetSize(), 2);

   // o2 should no longer be in the list or the hash table
   EXPECT_EQ(list.FindObject("item2"), nullptr);
   EXPECT_EQ(list.FindObject(o2), nullptr);

   // Other items should still be accessible
   EXPECT_EQ(list.FindObject("item1"), o1);
   EXPECT_EQ(list.FindObject("item3"), o3);
   EXPECT_EQ(list.First(), o1);
   EXPECT_EQ(list.Last(), o3);

   delete removed;
}

TEST(THashListTest, Rehash)
{
   THashList list(5);
   list.SetOwner(kTRUE);

   for (int i = 0; i < 20; ++i) {
      list.Add(new TNamed(Form("key_%d", i), Form("value_%d", i)));
   }

   EXPECT_EQ(list.GetSize(), 20);

   // Trigger rehash to a larger capacity
   list.Rehash(50);

   // All keys must still be found correctly after rehash
   for (int i = 0; i < 20; ++i) {
      TObject *obj = list.FindObject(Form("key_%d", i));
      EXPECT_NE(obj, nullptr);
      EXPECT_STREQ(obj->GetName(), Form("key_%d", i));
   }
}

TEST(THashListTest, ClearAndDelete)
{
   TrackNamed::fgCount = 0;

   // 1. Non-owner Clear(): removes from collection without deleting objects
   {
      THashList list;
      list.SetOwner(kFALSE);

      auto *o1 = new TrackNamed("k1", "v1");
      auto *o2 = new TrackNamed("k2", "v2");
      list.Add(o1);
      list.Add(o2);

      EXPECT_EQ(TrackNamed::fgCount, 2);
      EXPECT_EQ(list.GetSize(), 2);

      list.Clear();
      EXPECT_EQ(list.GetSize(), 0);
      EXPECT_EQ(list.FindObject("k1"), nullptr);
      EXPECT_EQ(TrackNamed::fgCount, 2);

      delete o1;
      delete o2;
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }

   // 2. Non-owner Delete(): explicitly deletes all objects
   {
      THashList list;
      list.SetOwner(kFALSE);

      list.Add(new TrackNamed("k1", "v1"));
      list.Add(new TrackNamed("k2", "v2"));

      EXPECT_EQ(TrackNamed::fgCount, 2);
      EXPECT_EQ(list.GetSize(), 2);

      list.Delete();
      EXPECT_EQ(list.GetSize(), 0);
      EXPECT_EQ(list.FindObject("k1"), nullptr);
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }

   // 3. Owner Clear(): automatically deletes all objects
   {
      THashList list;
      list.SetOwner(kTRUE);

      list.Add(new TrackNamed("k1", "v1"));
      list.Add(new TrackNamed("k2", "v2"));

      EXPECT_EQ(TrackNamed::fgCount, 2);
      EXPECT_EQ(list.GetSize(), 2);

      list.Clear();
      EXPECT_EQ(list.GetSize(), 0);
      EXPECT_EQ(list.FindObject("k1"), nullptr);
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }
}

TEST(THashListTest, Iteration)
{
   THashList list;
   list.SetOwner(kTRUE);

   list.Add(new TNamed("alpha", "1"));
   list.Add(new TNamed("beta", "2"));
   list.Add(new TNamed("gamma", "3"));

   int count = 0;
   TIter next(&list);
   TObject *obj = nullptr;
   while ((obj = next())) {
      count++;
   }

   EXPECT_EQ(count, 3);
}
