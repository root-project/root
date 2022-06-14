/// \file
/// \ingroup tutorial_vecops
/// \notebook -nodraw
/// In this tutorial we learn how elements of an RVec can be easily sorted and
/// selected.
///
/// \macro_code
/// \macro_output
///
/// \date August 2018
/// \author Stefan Wunsch

void vo004_SortAndSelect()
{
   // Because RVec implements an iterator, the class is fully compatible with
   // the sorting algorithms in the standard library.
   ROOT::RVecD v1{6., 4., 5.};
   ROOT::RVecD v2(v1);
   std::sort(v2.begin(), v2.end());
   std::cout << "Sort vector " << v1 << ": " << v2 << std::endl;

   // For convenience, ROOT implements helpers, e.g., to get a sorted copy of
   // an RVec ...
   auto v3 = Sort(v1);
   std::cout << "Sort vector " << v1 << ": " << v3 << std::endl;

   // ... or a reversed copy of an RVec.
   auto v4 = Reverse(v1);
   std::cout << "Reverse vector " << v1 << ": " << v4 << std::endl;

   // Helpers are provided to get the indices that sort the vector and to
   // select these indices from an RVec.
   auto i = Argsort(v1);
   std::cout << "Indices that sort the vector " << v1 << ": " << i << std::endl;

   ROOT::RVecD v5{9., 7., 8.};
   auto v6 = Take(v5, i);
   std::cout << "Sort vector " << v5 << " respective to the previously"
             << " determined indices: " << v6 << std::endl;

   // Take can also be used to get the first or last elements of an RVec.
   auto v7 = Take(v1, 2);
   auto v8 = Take(v1, -2);
   std::cout << "Take the two first and last elements of vector " << v1
             << ": " << v7 << ", " << v8 << std::endl;

   // Because the helpers return a copy of the input, you can chain the operations
   // conveniently.
   auto v9 = Reverse(Take(Sort(v1), -2));
   std::cout << "Sort the vector " << v1 << ", take the two last elements and "
             << "reverse the selection: " << v9 << std::endl;
}
