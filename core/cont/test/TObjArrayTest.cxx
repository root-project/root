#include "gtest/gtest.h"
#include "TObjArray.h"
#include "TObjString.h"

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
   EXPECT_EQ(array.At(2), nullptr); // Out of bounds
}

TEST(TObjArrayTest, AddFirstAndLast)
{
   TObjArray array;
   array.SetOwner(kTRUE);
   
   TObjString *s1 = new TObjString("first");
   TObjString *s2 = new TObjString("second");
   TObjString *s3 = new TObjString("third");
   
   array.AddFirst(s2); // Array: [s2]
   array.AddFirst(s1); // Array: [s1, s2]
   array.AddLast(s3);  // Array: [s1, s2, s3]
   
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
   array.AddAt(s2, 2); // Position 1 should be empty (nullptr)
   
   EXPECT_EQ(array.GetEntries(), 2);
   EXPECT_EQ(array.GetLast(), 2); // Highest index is 2
   
   EXPECT_EQ(array.At(0), s1);
   EXPECT_EQ(array.At(1), nullptr);
   EXPECT_EQ(array.At(2), s2);
}

TEST(TObjArrayTest, AddAtAndExpand)
{
   TObjArray array(2); // Initial size 2
   array.SetOwner(kTRUE);
   
   TObjString *s1 = new TObjString("element");
   
   // Add at index beyond current capacity
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
   
   // Should find the first free slot, which is index 1
   Int_t idx = array.AddAtFree(s3);
   EXPECT_EQ(idx, 1);
   EXPECT_EQ(array.At(1), s3);
   EXPECT_EQ(array.GetEntries(), 3);
}

TEST(TObjArrayTest, AddBeforeAndAfter)
{
   TObjArray array;
   array.SetOwner(kTRUE);
   
   TObjString *s1 = new TObjString("first");
   TObjString *s2 = new TObjString("second");
   TObjString *s3 = new TObjString("third");
   TObjString *s4 = new TObjString("fourth");
   
   array.Add(s1);
   array.Add(s3);
   
   array.AddBefore(s3, s2); // Array: [s1, s2, s3]
   array.AddAfter(s3, s4);  // Array: [s1, s2, s3, s4]
   
   EXPECT_EQ(array.GetEntries(), 4);
   EXPECT_EQ(array.At(0), s1);
   EXPECT_EQ(array.At(1), s2);
   EXPECT_EQ(array.At(2), s3);
   EXPECT_EQ(array.At(3), s4);
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
   
   // Remove leaves a "hole", doesn't shift elements in TObjArray
   EXPECT_EQ(array.At(0), s1);
   EXPECT_EQ(array.At(1), nullptr);
   EXPECT_EQ(array.At(2), s3);
   
   delete removed; // Clean up since array didn't delete it
}

TEST(TObjArrayTest, RemoveAt)
{
   TObjArray array;
   array.SetOwner(kTRUE);
   
   TObjString *s1 = new TObjString("first");
   TObjString *s2 = new TObjString("second");
   
   array.Add(s1);
   array.Add(s2);
   
   TObject *removed = array.RemoveAt(0);
   EXPECT_EQ(removed, s1);
   EXPECT_EQ(array.GetEntries(), 1);
   EXPECT_EQ(array.At(0), nullptr);
   EXPECT_EQ(array.At(1), s2);
   
   delete removed;
}

TEST(TObjArrayTest, ClearAndDelete)
{
   TObjArray array;
   
   TObjString *s1 = new TObjString("first");
   TObjString *s2 = new TObjString("second");
   
   array.Add(s1);
   array.Add(s2);
   
   // Clear doesn't delete elements unless SetOwner(kTRUE) is called or Option is "C"
   array.Clear();
   EXPECT_EQ(array.GetEntries(), 0);
   
   // Manually delete them since array wasn't owner
   delete s1;
   delete s2;
   
   TObjString *s3 = new TObjString("third");
   TObjString *s4 = new TObjString("fourth");
   
   array.Add(s3);
   array.Add(s4);
   
   // Delete forces deletion of elements regardless of SetOwner
   array.Delete();
   EXPECT_EQ(array.GetEntries(), 0);
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
