/// \file
/// \ingroup tutorial_vecops
/// \notebook -nodraw
/// In this tutorial we learn how an RVec can be sorted efficiently.
///
/// \macro_code
///
/// \date July 2018
/// \author Stefan Wunsch

using namespace ROOT::VecOps;

void vo004_Sorting()
{
   // Since RVec implements an iterator, the class is fully compatible with
   // the sorting algorithms in the standard library.
   RVec<double> v1{6., 4., 5.};
   RVec<double> v2(v1);
   std::sort(v2.begin(), v2.end());
   std::cout << "Sorting of vector " << v1 << ": " << v2 << std::endl;

   // Additionally, ROOT provides helpers to get the indices that sort the
   // vector and to select these indices from an RVec.
   RVec<double> v3{6., 4., 5.};
   auto i = Argsort(v3);
   std::cout << "Indices that sort the vector " << v3 << ": " << i << std::endl;

   RVec<double> v4{9., 7., 8.};
   auto v5 = Take(v4, i);
   std::cout << "Sorting of the vector " << v4 << " respective to the previously"
             << " determined indices: " << v5 << std::endl;
}
