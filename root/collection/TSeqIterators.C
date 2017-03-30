#include "ROOT/TSeq.hxx"

int TSeqIterators()
{
   ROOT::TSeqI seq(0, 6, 2);

   auto b = seq.begin();
   auto c = b + 1; // addition with integer
   if (c++ != ++b) // pre and post-increment
      return 1;
   if (seq.end() - seq.begin() != int(seq.size())) // difference of iterators, size
      return 2;
   if ((--c)-- != seq.begin() + 1) // pre-decrement and post-decrement
      return 3;
   b -= 1;       // compound assignment
   if (c != b--) // post decrement
      return 4;
   if (c <= b) // comparison operators
      return 5;
   if (b >= c) return 6;
   if (c[3] != *(seq.begin() + 3)) // subscript operator
      return 5;
   return 0;
}

int main()
{
   return TSeqIterators();
}
