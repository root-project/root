#include "gtest/gtest.h"
#include "TMap.h"
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

TEST(TMapTest, DefaultConstructor)
{
   TMap map;
   EXPECT_EQ(map.GetSize(), 0);
   EXPECT_TRUE(map.IsEmpty());
   EXPECT_GE(map.Capacity(), 0);
   EXPECT_EQ(map.GetValue("nonexistent"), nullptr);
   EXPECT_EQ(map.FindObject("nonexistent"), nullptr);
}

TEST(TMapTest, AddAndLookup)
{
   TMap map;
   map.SetOwnerKeyValue(kTRUE, kTRUE);

   EXPECT_TRUE(map.IsOwner());
   EXPECT_TRUE(map.IsOwnerValue());

   auto *k1 = new TNamed("key1", "k1");
   auto *v1 = new TNamed("val1", "v1");
   auto *k2 = new TNamed("key2", "k2");
   auto *v2 = new TNamed("val2", "v2");

   map.Add(k1, v1);
   map.Add(k2, v2);

   EXPECT_EQ(map.GetSize(), 2);
   EXPECT_FALSE(map.IsEmpty());
   EXPECT_NE(map.GetTable(), nullptr);
   EXPECT_EQ(map.GetTable()->GetSize(), 2);

   // Lookup by string
   EXPECT_EQ(map.GetValue("key1"), v1);
   EXPECT_EQ(map.GetValue("key2"), v2);
   EXPECT_EQ(map.GetValue("key3"), nullptr);

   // Lookup by object pointer
   EXPECT_EQ(map.GetValue(k1), v1);
   EXPECT_EQ(map.GetValue(k2), v2);

   TNamed dummy("key3", "");
   EXPECT_EQ(map.GetValue(&dummy), nullptr);

   // Function call operator operator()
   EXPECT_EQ(map("key1"), v1);
   EXPECT_EQ(map("key2"), v2);
   EXPECT_EQ(map(k1), v1);
   EXPECT_EQ(map(k2), v2);

   // FindObject returns TPair*
   auto *p1 = static_cast<TPair *>(map.FindObject("key1"));
   ASSERT_NE(p1, nullptr);
   EXPECT_EQ(p1->Key(), k1);
   EXPECT_EQ(p1->Value(), v1);

   auto *p2 = static_cast<TPair *>(map.FindObject(k2));
   ASSERT_NE(p2, nullptr);
   EXPECT_EQ(p2->Key(), k2);
   EXPECT_EQ(p2->Value(), v2);

   // Collisions check
   EXPECT_GE(map.Collisions("key1"), 1);
   EXPECT_GE(map.Collisions(k2), 1);
   EXPECT_EQ(map.Collisions("nonexistent"), 0);
}

TEST(TMapTest, HashCollision)
{
   // Use a small fixed capacity table without automatic rehash so collisions are guaranteed
   TMap map(3, 0);
   map.SetOwnerKeyValue(kTRUE, kTRUE);

   // "k0" (Hash=27504, 27504 % 3 == 0) and "k3" (Hash=27507, 27507 % 3 == 0) collide in slot 0
   auto *k1 = new TNamed("k0", "First");
   auto *v1 = new TNamed("v0", "Value 0");
   auto *k2 = new TNamed("k3", "Second");
   auto *v2 = new TNamed("v3", "Value 3");

   map.Add(k1, v1);
   map.Add(k2, v2);

   EXPECT_EQ(map.GetSize(), 2);

   // Both keys hash to the same bucket (slot 0) in the 3-slot table, creating a deliberate collision
   EXPECT_EQ(map.Collisions("k0"), 2);
   EXPECT_EQ(map.Collisions("k3"), 2);
   EXPECT_EQ(map.Collisions(k1), 2);
   EXPECT_EQ(map.Collisions(k2), 2);

   // Despite hash collision, lookups should correctly resolve both distinct items
   EXPECT_EQ(map.GetValue("k0"), v1);
   EXPECT_EQ(map.GetValue("k3"), v2);
   EXPECT_EQ(map.GetValue(k1), v1);
   EXPECT_EQ(map.GetValue(k2), v2);

   // Removing one collided item should leave the other intact and reduce collision count
   TPair *entry = map.RemoveEntry(k1);
   ASSERT_NE(entry, nullptr);
   EXPECT_EQ(entry->Key(), k1);
   EXPECT_EQ(entry->Value(), v1);
   delete entry;
   delete k1;
   delete v1;

   EXPECT_EQ(map.GetSize(), 1);
   EXPECT_EQ(map.Collisions("k3"), 1);
   EXPECT_EQ(map.GetValue("k0"), nullptr);
   EXPECT_EQ(map.GetValue("k3"), v2);
}

TEST(TMapTest, Remove)
{
   TMap map;

   auto *k1 = new TNamed("k1", "");
   auto *v1 = new TNamed("v1", "");
   auto *k2 = new TNamed("k2", "");
   auto *v2 = new TNamed("v2", "");

   map.Add(k1, v1);
   map.Add(k2, v2);

   // Remove removes entry from map and returns the key
   TObject *removedKey = map.Remove(k1);
   EXPECT_EQ(removedKey, k1);
   EXPECT_EQ(map.GetSize(), 1);
   EXPECT_EQ(map.GetValue("k1"), nullptr);
   EXPECT_EQ(map.GetValue("k2"), v2);

   map.Clear();
   delete k1;
   delete v1;
   delete k2;
   delete v2;
}

TEST(TMapTest, RemoveEntry)
{
   TMap map;

   auto *k1 = new TNamed("k1", "");
   auto *v1 = new TNamed("v1", "");

   map.Add(k1, v1);

   TPair *entry = map.RemoveEntry(k1);
   ASSERT_NE(entry, nullptr);
   EXPECT_EQ(entry->Key(), k1);
   EXPECT_EQ(entry->Value(), v1);
   EXPECT_EQ(map.GetSize(), 0);
   EXPECT_EQ(map.GetValue("k1"), nullptr);

   delete entry;
   delete k1;
   delete v1;
}

TEST(TMapTest, DeleteEntry)
{
   TrackNamed::fgCount = 0;

   {
      TMap map;
      map.SetOwnerKeyValue(kTRUE, kTRUE);

      auto *k1 = new TrackNamed("k1", "");
      auto *v1 = new TrackNamed("v1", "");
      auto *k2 = new TrackNamed("k2", "");
      auto *v2 = new TrackNamed("v2", "");

      map.Add(k1, v1);
      map.Add(k2, v2);

      EXPECT_EQ(TrackNamed::fgCount, 4);

      // DeleteEntry deletes both key and value because ownership is enabled
      Bool_t deleted = map.DeleteEntry(k1);
      EXPECT_TRUE(deleted);
      EXPECT_EQ(map.GetSize(), 1);
      EXPECT_EQ(map.GetValue("k1"), nullptr);
      EXPECT_EQ(TrackNamed::fgCount, 2);

      // Try deleting non-existent key
      TNamed nonexistent("dummy", "");
      EXPECT_FALSE(map.DeleteEntry(&nonexistent));
   }

   EXPECT_EQ(TrackNamed::fgCount, 0);
}

TEST(TMapTest, Rehash)
{
   TMap map(5);
   map.SetOwnerKeyValue(kTRUE, kTRUE);

   for (int i = 0; i < 20; ++i) {
      map.Add(new TNamed(Form("key_%d", i), ""), new TNamed(Form("val_%d", i), ""));
   }

   EXPECT_EQ(map.GetSize(), 20);

   map.Rehash(50);
   EXPECT_GE(map.Capacity(), 50);

   for (int i = 0; i < 20; ++i) {
      TObject *val = map.GetValue(Form("key_%d", i));
      ASSERT_NE(val, nullptr);
      EXPECT_STREQ(val->GetName(), Form("val_%d", i));
   }
}

TEST(TMapTest, OwnershipIndependent)
{
   TrackNamed::fgCount = 0;

   // 1. Only keys owned: SetOwner(kTRUE), SetOwnerValue(kFALSE)
   {
      TMap map;
      map.SetOwner(kTRUE);
      map.SetOwnerValue(kFALSE);
      EXPECT_TRUE(map.IsOwner());
      EXPECT_FALSE(map.IsOwnerValue());

      auto *k = new TrackNamed("k_only", "");
      auto *v = new TrackNamed("v_only", "");
      map.Add(k, v);

      EXPECT_EQ(TrackNamed::fgCount, 2);

      map.Clear(); // deletes key, leaves value alive
      EXPECT_EQ(TrackNamed::fgCount, 1);

      delete v;
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }

   // 2. Only values owned: SetOwner(kFALSE), SetOwnerValue(kTRUE)
   {
      TMap map;
      map.SetOwner(kFALSE);
      map.SetOwnerValue(kTRUE);
      EXPECT_FALSE(map.IsOwner());
      EXPECT_TRUE(map.IsOwnerValue());

      auto *k = new TrackNamed("k_only", "");
      auto *v = new TrackNamed("v_only", "");
      map.Add(k, v);

      EXPECT_EQ(TrackNamed::fgCount, 2);

      map.Clear(); // deletes value, leaves key alive
      EXPECT_EQ(TrackNamed::fgCount, 1);

      delete k;
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }

   // 3. Removal with value ownership enabled deletes value
   {
      TMap map;
      map.SetOwnerValue(kTRUE);

      auto *k = new TrackNamed("k_rm", "");
      auto *v = new TrackNamed("v_rm", "");
      map.Add(k, v);

      EXPECT_EQ(TrackNamed::fgCount, 2);

      TObject *retKey = map.Remove(k);
      EXPECT_EQ(retKey, k);
      // Remove() deletes value if IsOwnerValue() is true
      EXPECT_EQ(TrackNamed::fgCount, 1);

      delete k;
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }

   // 4. DeleteEntry with independent ownership
   {
      TMap map;
      map.SetOwner(kTRUE);
      map.SetOwnerValue(kFALSE);

      auto *k = new TrackNamed("k_del", "");
      auto *v = new TrackNamed("v_del", "");
      map.Add(k, v);

      EXPECT_EQ(TrackNamed::fgCount, 2);

      EXPECT_TRUE(map.DeleteEntry(k)); // deletes key, value stays alive
      EXPECT_EQ(TrackNamed::fgCount, 1);

      delete v;
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }
}

TEST(TMapTest, ClearAndDelete)
{
   TrackNamed::fgCount = 0;

   // 1. Non-owner Clear(): removes entries from map without deleting keys or values
   {
      TMap map;
      auto *k1 = new TrackNamed("k1", "");
      auto *v1 = new TrackNamed("v1", "");
      map.Add(k1, v1);

      EXPECT_EQ(TrackNamed::fgCount, 2);
      EXPECT_EQ(map.GetSize(), 1);

      map.Clear();
      EXPECT_EQ(map.GetSize(), 0);
      EXPECT_EQ(TrackNamed::fgCount, 2);

      delete k1;
      delete v1;
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }

   // 2. DeleteKeys(): deletes only key objects
   {
      TMap map;
      auto *k1 = new TrackNamed("k1", "");
      auto *v1 = new TrackNamed("v1", "");
      map.Add(k1, v1);

      EXPECT_EQ(TrackNamed::fgCount, 2);

      map.DeleteKeys();
      EXPECT_EQ(map.GetSize(), 0);
      EXPECT_EQ(TrackNamed::fgCount, 1); // Value still alive

      delete v1;
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }

   // 3. DeleteValues(): deletes only value objects
   {
      TMap map;
      auto *k1 = new TrackNamed("k1", "");
      auto *v1 = new TrackNamed("v1", "");
      map.Add(k1, v1);

      EXPECT_EQ(TrackNamed::fgCount, 2);

      map.DeleteValues();
      EXPECT_EQ(map.GetSize(), 0);
      EXPECT_EQ(TrackNamed::fgCount, 1); // Key still alive

      delete k1;
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }

   // 4. DeleteAll(): deletes both key and value objects
   {
      TMap map;
      map.Add(new TrackNamed("k1", ""), new TrackNamed("v1", ""));
      map.Add(new TrackNamed("k2", ""), new TrackNamed("v2", ""));

      EXPECT_EQ(TrackNamed::fgCount, 4);

      map.DeleteAll();
      EXPECT_EQ(map.GetSize(), 0);
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }

   // 5. Owner Clear(): deletes both keys and values when SetOwnerKeyValue(kTRUE, kTRUE) is set
   {
      TMap map;
      map.SetOwnerKeyValue(kTRUE, kTRUE);
      map.Add(new TrackNamed("k1", ""), new TrackNamed("v1", ""));
      map.Add(new TrackNamed("k2", ""), new TrackNamed("v2", ""));

      EXPECT_EQ(TrackNamed::fgCount, 4);

      map.Clear();
      EXPECT_EQ(map.GetSize(), 0);
      EXPECT_EQ(TrackNamed::fgCount, 0);
   }
}

TEST(TMapTest, Iteration)
{
   TMap map;
   map.SetOwnerKeyValue(kTRUE, kTRUE);

   map.Add(new TNamed("alpha", "1"), new TNamed("valA", ""));
   map.Add(new TNamed("beta", "2"), new TNamed("valB", ""));
   map.Add(new TNamed("gamma", "3"), new TNamed("valC", ""));

   int count = 0;
   TIter next(&map);
   TObject *key = nullptr;
   while ((key = next())) {
      TObject *val = map.GetValue(key);
      EXPECT_NE(val, nullptr);
      count++;
   }

   EXPECT_EQ(count, 3);
}
