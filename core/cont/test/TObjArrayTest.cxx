#include "gtest/gtest.h"
#include "TObjArray.h"
#include "TObjString.h"

namespace {
class TrackObject : public TObject {
public:
   static int fgCount;
   TrackObject() { fgCount++; }
   ~TrackObject() override { fgCount--; }
};
int TrackObject::fgCount = 0;
} // namespace

TEST(TObjArrayTest, DefaultConstructor)
{
   TObjArray array;
   EXPECT_EQ(array.GetEntries(), 0);
   EXPECT_EQ(array.GetEntriesFast(), 0);
   EXPECT_EQ(array.GetLast(), -1);
   EXPECT_TRUE(array.IsEmpty());
}

TEST(TObjArrayTest, AddAndAt)
{
   TObjArray array;
   array.SetOwner(kTRUE);

   TObjString *s1 = new TObjString("first");
   TObjString *s2 = new TObjString("second");

   array.Add(s1);
   array.Add(s2);

   EXPECT_EQ(array.GetEntries(), 2);
   EXPECT_EQ(array.GetLast(), 1);
   EXPECT_FALSE(array.IsEmpty());

   EXPECT_EQ(array.At(0), s1);
   EXPECT_EQ(array.At(1), s2);
   EXPECT_EQ(array.At(2), nullptr);
}

TEST(TObjArrayTest, AddFirstAndLast)
{
   TObjArray array;
   array.SetOwner(kTRUE);

   TObjString *s1 = new TObjString("first");
   TObjString *s2 = new TObjString("second");
   TObjString *s3 = new TObjString("third");

   array.AddFirst(s2);
   array.AddFirst(s1);
   array.AddLast(s3);

   EXPECT_EQ(array.GetEntries(), 3);
   EXPECT_EQ(array.At(0), s1);
   EXPECT_EQ(array.At(1), s2);
   EXPECT_EQ(array.At(2), s3);
}

TEST(TObjArrayTest, AddAt)
{
   TObjArray array;
   array.SetOwner(kTRUE);

   TObjString *s1 = new TObjString("first");
   TObjString *s2 = new TObjString("second");

   array.AddAt(s1, 0);
   array.AddAt(s2, 2);

   EXPECT_EQ(array.GetEntries(), 2);
   EXPECT_EQ(array.GetLast(), 2);

   EXPECT_EQ(array.At(0), s1);
   EXPECT_EQ(array.At(1), nullptr);
   EXPECT_EQ(array.At(2), s2);
}

TEST(TObjArrayTest, AddAtAndExpand)
{
   TObjArray array(2);
   array.SetOwner(kTRUE);

   TObjString *s1 = new TObjString("element");

   array.AddAtAndExpand(s1, 5);

   EXPECT_EQ(array.GetEntries(), 1);
   EXPECT_EQ(array.GetLast(), 5);
   EXPECT_EQ(array.At(5), s1);
}

TEST(TObjArrayTest, AddAtFree)
{
   TObjArray array;
   array.SetOwner(kTRUE);

   TObjString *s1 = new TObjString("first");
   TObjString *s2 = new TObjString("second");
   TObjString *s3 = new TObjString("third");

   array.AddAt(s1, 0);
   array.AddAt(s2, 2);

   Int_t idx = array.AddAtFree(s3);
   EXPECT_EQ(idx, 1);
   EXPECT_EQ(array.At(1), s3);
   EXPECT_EQ(array.GetEntries(), 3);
}

TEST(TObjArrayTest, AddBeforeAndAfter)
{
   TObjArray array;
   array.SetOwner(kTRUE);

   TObjString *s2 = new TObjString("second");
   array.AddAt(s2, 2);

   TObjString *s1 = new TObjString("first");
   array.AddBefore(s2, s1);

   TObjString *s3 = new TObjString("third");
   array.AddAfter(s2, s3);

   EXPECT_EQ(array.GetEntries(), 3);
   EXPECT_EQ(array.At(1), s1);
   EXPECT_EQ(array.At(2), s2);
   EXPECT_EQ(array.At(3), s3);
}

TEST(TObjArrayTest, Remove)
{
   TObjArray array;
   array.SetOwner(kTRUE);

   TObjString *s1 = new TObjString("first");
   TObjString *s2 = new TObjString("second");
   TObjString *s3 = new TObjString("third");

   array.Add(s1);
   array.Add(s2);
   array.Add(s3);

   TObject *removed = array.Remove(s2);
   EXPECT_EQ(removed, s2);
   EXPECT_EQ(array.GetEntries(), 2);

   EXPECT_EQ(array.At(0), s1);
   EXPECT_EQ(array.At(1), nullptr);
   EXPECT_EQ(array.At(2), s3);

   delete removed;
}

TEST(TObjArrayTest, RemoveAt)
{
   // Test removing middle element
   {
      TObjArray array;
      array.SetOwner(kTRUE);

      TObjString *s1 = new TObjString("first");
      TObjString *s2 = new TObjString("second");
      TObjString *s3 = new TObjString("third");

      array.Add(s1);
      array.Add(s2);
      array.Add(s3);
      EXPECT_EQ(array.GetLast(), 2);

      TObject *removed = array.RemoveAt(1);
      EXPECT_EQ(removed, s2);
      EXPECT_EQ(array.GetEntries(), 2);
      EXPECT_EQ(array.At(0), s1);
      EXPECT_EQ(array.At(1), nullptr);
      EXPECT_EQ(array.At(2), s3);
      EXPECT_EQ(array.GetLast(), 2);

      delete removed;
   }

   // Test removing last element updates GetLast()
   {
      TObjArray array;
      array.SetOwner(kTRUE);

      TObjString *s1 = new TObjString("first");
      TObjString *s2 = new TObjString("second");
      TObjString *s3 = new TObjString("third");

      array.Add(s1);
      array.Add(s2);
      array.Add(s3);
      EXPECT_EQ(array.GetLast(), 2);

      TObject *removed = array.RemoveAt(2);
      EXPECT_EQ(removed, s3);
      EXPECT_EQ(array.GetEntries(), 2);
      EXPECT_EQ(array.At(0), s1);
      EXPECT_EQ(array.At(1), s2);
      EXPECT_EQ(array.At(2), nullptr);
      EXPECT_EQ(array.GetLast(), 1);

      delete removed;
   }
}

TEST(TObjArrayTest, Compress)
{
   TObjArray array;
   array.SetOwner(kTRUE);

   TObjString *s1 = new TObjString("first");
   TObjString *s2 = new TObjString("second");
   TObjString *s3 = new TObjString("third");

   array.Add(s1);
   array.Add(s2);
   array.Add(s3);

   // Remove the middle element, creating a gap: [s1, nullptr, s3]
   TObject *removed = array.RemoveAt(1);
   delete removed;

   EXPECT_EQ(array.GetEntries(), 2);
   EXPECT_EQ(array.At(0), s1);
   EXPECT_EQ(array.At(1), nullptr);
   EXPECT_EQ(array.At(2), s3);
   EXPECT_EQ(array.GetLast(), 2);

   // Compress closes the gap: [s1, s3, nullptr]
   array.Compress();

   EXPECT_EQ(array.GetEntries(), 2);
   EXPECT_EQ(array.At(0), s1);
   EXPECT_EQ(array.At(1), s3);
   EXPECT_EQ(array.At(2), nullptr);
   EXPECT_EQ(array.GetLast(), 1);
}

TEST(TObjArrayTest, ClearAndDelete)
{
   TrackObject::fgCount = 0;

   // 1. Non-owner Clear(): removes pointers but does NOT delete objects
   {
      TObjArray array;
      array.SetOwner(kFALSE);

      auto *o1 = new TrackObject();
      auto *o2 = new TrackObject();
      array.Add(o1);
      array.Add(o2);

      EXPECT_EQ(TrackObject::fgCount, 2);
      EXPECT_EQ(array.GetEntries(), 2);

      array.Clear();
      EXPECT_EQ(array.GetEntries(), 0);
      EXPECT_EQ(TrackObject::fgCount, 2);

      delete o1;
      delete o2;
      EXPECT_EQ(TrackObject::fgCount, 0);
   }

   // 2. Non-owner Delete(): explicitly deletes all heap objects
   {
      TObjArray array;
      array.SetOwner(kFALSE);

      array.Add(new TrackObject());
      array.Add(new TrackObject());

      EXPECT_EQ(TrackObject::fgCount, 2);
      EXPECT_EQ(array.GetEntries(), 2);

      array.Delete();
      EXPECT_EQ(array.GetEntries(), 0);
      EXPECT_EQ(TrackObject::fgCount, 0);
   }

   // 3. Owner Clear(): delegates to Delete(), deleting all objects
   {
      TObjArray array;
      array.SetOwner(kTRUE);

      array.Add(new TrackObject());
      array.Add(new TrackObject());

      EXPECT_EQ(TrackObject::fgCount, 2);
      EXPECT_EQ(array.GetEntries(), 2);

      array.Clear();
      EXPECT_EQ(array.GetEntries(), 0);
      EXPECT_EQ(TrackObject::fgCount, 0);
   }
}

TEST(TObjArrayTest, Iteration)
{
   TObjArray array;
   array.SetOwner(kTRUE);

   array.Add(new TObjString("A"));
   array.Add(new TObjString("B"));
   array.Add(new TObjString("C"));

   Int_t count = 0;
   TIter next(&array);
   TObject *obj;
   while ((obj = next())) {
      count++;
   }

   EXPECT_EQ(count, 3);
}
