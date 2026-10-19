#include "gtest/gtest.h"
#include "TOrdCollection.h"
#include "TNamed.h"
#include "TObjString.h"

#include <string>
#include <vector>

TEST(TOrdCollectionTest, DefaultConstructor)
{
   TOrdCollection col;
   EXPECT_EQ(0, col.GetSize());
   EXPECT_TRUE(col.IsEmpty());
   EXPECT_EQ(nullptr, col.First());
   EXPECT_EQ(nullptr, col.Last());
   EXPECT_EQ(nullptr, col.At(0));
   EXPECT_EQ(nullptr, col.At(-1));
   EXPECT_EQ(nullptr, col.At(5));
   EXPECT_FALSE(col.IsOwner());
}

TEST(TOrdCollectionTest, CustomCapacity)
{
   TOrdCollection col(16);
   EXPECT_EQ(0, col.GetSize());
   EXPECT_TRUE(col.IsEmpty());

   TObjString obj1("val1");
   col.Add(&obj1);
   EXPECT_EQ(1, col.GetSize());
   EXPECT_EQ(&obj1, col.At(0));
}

TEST(TOrdCollectionTest, AddAndAt)
{
   TOrdCollection col;
   TObjString a("A");
   TObjString b("B");
   TObjString c("C");
   TObjString d("D");

   col.AddLast(&b);
   col.AddFirst(&a);
   col.AddLast(&d);
   col.AddAt(&c, 2);

   EXPECT_EQ(4, col.GetSize());
   EXPECT_EQ(&a, col.At(0));
   EXPECT_EQ(&b, col.At(1));
   EXPECT_EQ(&c, col.At(2));
   EXPECT_EQ(&d, col.At(3));

   EXPECT_EQ(&a, col.First());
   EXPECT_EQ(&d, col.Last());

   // Adding beyond current size clamps to AddLast
   TObjString e("E");
   col.AddAt(&e, 100);
   EXPECT_EQ(5, col.GetSize());
   EXPECT_EQ(&e, col.Last());
   EXPECT_EQ(&e, col.At(4));
}

TEST(TOrdCollectionTest, AddBeforeAndAfter)
{
   TOrdCollection col;
   TObjString first("first");
   TObjString middle("middle");
   TObjString last("last");

   col.Add(&first);
   col.Add(&last);

   col.AddAfter(&first, &middle);
   EXPECT_EQ(3, col.GetSize());
   EXPECT_EQ(&first, col.At(0));
   EXPECT_EQ(&middle, col.At(1));
   EXPECT_EQ(&last, col.At(2));

   TObjString newFirst("newFirst");
   col.AddBefore(&first, &newFirst);
   EXPECT_EQ(4, col.GetSize());
   EXPECT_EQ(&newFirst, col.At(0));
   EXPECT_EQ(&first, col.At(1));

   // Passing nullptr to AddBefore prepends (AddFirst)
   TObjString nullBefore("nullBefore");
   col.AddBefore(nullptr, &nullBefore);
   EXPECT_EQ(&nullBefore, col.First());

   // Passing nullptr to AddAfter appends (AddLast)
   TObjString nullAfter("nullAfter");
   col.AddAfter(nullptr, &nullAfter);
   EXPECT_EQ(&nullAfter, col.Last());
}

TEST(TOrdCollectionTest, Navigation)
{
   TOrdCollection col;
   TObjString a("A");
   TObjString b("B");
   TObjString c("C");

   col.Add(&a);
   col.Add(&b);
   col.Add(&c);

   EXPECT_EQ(&b, col.After(&a));
   EXPECT_EQ(&c, col.After(&b));
   EXPECT_EQ(nullptr, col.After(&c));
   EXPECT_EQ(nullptr, col.After(nullptr));

   EXPECT_EQ(nullptr, col.Before(&a));
   EXPECT_EQ(&a, col.Before(&b));
   EXPECT_EQ(&b, col.Before(&c));
   EXPECT_EQ(nullptr, col.Before(nullptr));

   EXPECT_EQ(0, col.IndexOf(&a));
   EXPECT_EQ(1, col.IndexOf(&b));
   EXPECT_EQ(2, col.IndexOf(&c));

   TObjString notInCol("X");
   EXPECT_EQ(-1, col.IndexOf(&notInCol));
   EXPECT_EQ(nullptr, col.Before(&notInCol));
   EXPECT_EQ(nullptr, col.After(&notInCol));
}

TEST(TOrdCollectionTest, PutAt)
{
   TOrdCollection col;
   TObjString a("A");
   TObjString b("B");
   TObjString repl("Replacement");

   col.Add(&a);
   col.Add(&b);

   EXPECT_EQ(&b, col.At(1));
   col.PutAt(&repl, 1);
   EXPECT_EQ(2, col.GetSize());
   EXPECT_EQ(&repl, col.At(1));
}

TEST(TOrdCollectionTest, RemoveAndRemoveAt)
{
   TOrdCollection col;
   const int n = 10;
   std::vector<TObjString> items;
   items.reserve(n);
   for (int i = 0; i < n; ++i) {
      items.emplace_back(std::to_string(i).c_str());
      col.Add(&items.back());
   }
   EXPECT_EQ(n, col.GetSize());

   // Remove middle element
   TObject *removedMiddle = col.RemoveAt(4);
   EXPECT_EQ(&items[4], removedMiddle);
   EXPECT_EQ(n - 1, col.GetSize());
   EXPECT_EQ(&items[5], col.At(4));

   // Remove first element
   TObject *removedFirst = col.RemoveAt(0);
   EXPECT_EQ(&items[0], removedFirst);
   EXPECT_EQ(n - 2, col.GetSize());
   EXPECT_EQ(&items[1], col.First());

   // Remove by pointer
   TObject *removedLast = col.Remove(&items[n - 1]);
   EXPECT_EQ(&items[n - 1], removedLast);
   EXPECT_EQ(n - 3, col.GetSize());
   EXPECT_EQ(&items[n - 2], col.Last());

   // Remove non-existent object
   TObjString outside("outside");
   EXPECT_EQ(nullptr, col.Remove(&outside));
   EXPECT_EQ(nullptr, col.Remove(nullptr));
}

TEST(TOrdCollectionTest, SortAndBinarySearch)
{
   TOrdCollection col;
   TNamed n3("gamma", "3");
   TNamed n1("alpha", "1");
   TNamed n4("delta", "4");
   TNamed n2("beta", "2");

   col.Add(&n3);
   col.Add(&n1);
   col.Add(&n4);
   col.Add(&n2);

   EXPECT_EQ(&n3, col.At(0));

   col.Sort();

   EXPECT_EQ(4, col.GetSize());
   EXPECT_EQ(&n1, col.At(0)); // alpha
   EXPECT_EQ(&n2, col.At(1)); // beta
   EXPECT_EQ(&n4, col.At(2)); // delta
   EXPECT_EQ(&n3, col.At(3)); // gamma

   EXPECT_EQ(0, col.BinarySearch(&n1));
   EXPECT_EQ(1, col.BinarySearch(&n2));
   EXPECT_EQ(2, col.BinarySearch(&n4));
   EXPECT_EQ(3, col.BinarySearch(&n3));

   TNamed notFound("epsilon", "5");
   EXPECT_EQ(-1, col.BinarySearch(&notFound));
   EXPECT_EQ(-1, col.BinarySearch(nullptr));
}

namespace {
class TrackedObj : public TObject {
public:
   static int sAliveCount;
   TrackedObj() { ++sAliveCount; }
   ~TrackedObj() override { --sAliveCount; }
};
int TrackedObj::sAliveCount = 0;
} // namespace

TEST(TOrdCollectionTest, OwnershipAndClearDelete)
{
   TrackedObj::sAliveCount = 0;

   // Non-owning collection
   {
      TOrdCollection col;
      auto *obj1 = new TrackedObj();
      auto *obj2 = new TrackedObj();
      col.Add(obj1);
      col.Add(obj2);
      EXPECT_EQ(2, TrackedObj::sAliveCount);

      col.Clear();
      EXPECT_EQ(0, col.GetSize());
      EXPECT_EQ(2, TrackedObj::sAliveCount);

      col.Add(obj1);
      col.Add(obj2);
      col.Delete();
      EXPECT_EQ(0, col.GetSize());
      EXPECT_EQ(0, TrackedObj::sAliveCount);
   }

   // Owning collection via SetOwner(kTRUE)
   TrackedObj::sAliveCount = 0;
   {
      TOrdCollection col;
      col.SetOwner(kTRUE);
      EXPECT_TRUE(col.IsOwner());

      col.Add(new TrackedObj());
      col.Add(new TrackedObj());
      EXPECT_EQ(2, TrackedObj::sAliveCount);

      col.Clear();
      EXPECT_EQ(0, col.GetSize());
      EXPECT_EQ(0, TrackedObj::sAliveCount);

      col.Add(new TrackedObj());
      col.Add(new TrackedObj());
      EXPECT_EQ(2, TrackedObj::sAliveCount);
   }
   // Destructor cleans up owned objects
   EXPECT_EQ(0, TrackedObj::sAliveCount);
}

TEST(TOrdCollectionTest, Iteration)
{
   TOrdCollection col;
   TObjString a("A");
   TObjString b("B");
   TObjString c("C");

   col.Add(&a);
   col.Add(&b);
   col.Add(&c);

   // Forward iteration with TIter
   {
      TIter next(&col);
      EXPECT_EQ(&a, next());
      EXPECT_EQ(&b, next());
      EXPECT_EQ(&c, next());
      EXPECT_EQ(nullptr, next());

      next.Reset();
      EXPECT_EQ(&a, next());
   }

   // Backward iteration
   {
      TIter next(&col, kIterBackward);
      EXPECT_EQ(&c, next());
      EXPECT_EQ(&b, next());
      EXPECT_EQ(&a, next());
      EXPECT_EQ(nullptr, next());
   }

   // TOrdCollectionIter direct usage
   {
      TOrdCollectionIter iter(&col);
      EXPECT_EQ(&col, iter.GetCollection());
      EXPECT_EQ(&a, iter.Next());
      EXPECT_EQ(&a, *iter);
      EXPECT_EQ(&b, iter.Next());
      EXPECT_EQ(&b, *iter);
      EXPECT_EQ(&c, iter.Next());
      EXPECT_EQ(&c, *iter);
      EXPECT_EQ(nullptr, iter.Next());
   }
}
