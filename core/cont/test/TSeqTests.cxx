#include "gtest/gtest.h"
#include "ROOT/TSeq.hxx"

TEST(TSeqIterators, Containers)
{
   ROOT::TSeqI seq(0, 6, 2);
   auto b = seq.begin();
   auto c = b + 1;                                      // addition with integer
   ASSERT_TRUE(c++ == ++b);                             // pre and post-increment
   ASSERT_EQ(seq.end() - seq.begin(), int(seq.size())); // difference of iterators, size
   ASSERT_TRUE((--c)-- == seq.begin() + 1);             // pre-decrement and post-decrement
   b -= 1;                                              // compound assignment
   ASSERT_TRUE(c == b--);                               // post decrement
   // comparison operators
   ASSERT_GE(c, b);
   ASSERT_LE(b, c);
   ASSERT_EQ(c[3], *(seq.begin() + 3)); // subscript operator
}
