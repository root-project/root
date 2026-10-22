#include "gtest/gtest.h"
#include "TList.h"
#include "TObjString.h"

TEST(TList, BasicAddRemove)
{
   TList list;
   EXPECT_EQ(0, list.GetSize());

   TObjString *s1 = new TObjString("s1");
   TObjString *s2 = new TObjString("s2");

   list.Add(s1);
   EXPECT_EQ(1, list.GetSize());
   EXPECT_EQ(s1, list.First());
   EXPECT_EQ(s1, list.Last());

   list.Add(s2);
   EXPECT_EQ(2, list.GetSize());
   EXPECT_EQ(s1, list.First());
   EXPECT_EQ(s2, list.Last());

   TObject *removed = list.Remove(s1);
   EXPECT_EQ(s1, removed);
   EXPECT_EQ(1, list.GetSize());
   EXPECT_EQ(s2, list.First());
   EXPECT_EQ(s2, list.Last());

   delete s1;

   list.Clear();
   EXPECT_EQ(0, list.GetSize());
   delete s2;
}

TEST(TList, Insertions)
{
   TList list;
   list.SetOwner(kTRUE);

   TObjString *s1 = new TObjString("s1");
   TObjString *s2 = new TObjString("s2");
   TObjString *s3 = new TObjString("s3");

   list.AddFirst(s2);
   list.AddFirst(s1);
   list.AddLast(s3);

   EXPECT_EQ(3, list.GetSize());
   EXPECT_EQ(s1, list.At(0));
   EXPECT_EQ(s2, list.At(1));
   EXPECT_EQ(s3, list.At(2));

   TObjString *s1_5 = new TObjString("s1.5");
   list.AddAfter(s1, s1_5);
   EXPECT_EQ(s1_5, list.At(1));
   EXPECT_EQ(s2, list.At(2));

   TObjString *s2_5 = new TObjString("s2.5");
   list.AddBefore(s3, s2_5);
   EXPECT_EQ(s2_5, list.At(3));
   EXPECT_EQ(s3, list.At(4));
}

TEST(TList, FindObject)
{
   TList list;
   list.SetOwner(kTRUE);

   TObjString *s1 = new TObjString("s1");
   TObjString *s2 = new TObjString("s2");
   list.Add(s1);
   list.Add(s2);

   EXPECT_EQ(s1, list.FindObject("s1"));
   EXPECT_EQ(s2, list.FindObject("s2"));
   EXPECT_EQ(nullptr, list.FindObject("nonexistent"));
}

TEST(TList, Sort)
{
   TList list;
   list.SetOwner(kTRUE);

   TObjString *s1 = new TObjString("C");
   TObjString *s2 = new TObjString("A");
   TObjString *s3 = new TObjString("B");

   list.Add(s1);
   list.Add(s2);
   list.Add(s3);

   list.Sort();

   EXPECT_EQ(s2, list.At(0)); // A
   EXPECT_EQ(s3, list.At(1)); // B
   EXPECT_EQ(s1, list.At(2)); // C
}
