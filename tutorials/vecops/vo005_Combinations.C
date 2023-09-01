/// \file
/// \ingroup tutorial_vecops
/// \notebook -nodraw
/// In this tutorial we learn how combinations of RVecs can be built.
///
/// \macro_code
/// \macro_output
///
/// \date August 2018
/// \author Stefan Wunsch

void vo005_Combinations()
{
   // The starting point are two collections and we want to calculate the result
   // of combinations of the elements.
   ROOT::RVecD v1{1., 2., 3.};
   ROOT::RVecD v2{-4., -5.};

   // To get the indices, which result in all combinations, you can call the
   // following helper.
   // Note that you can also pass the size of the vectors directly.
   auto idx = Combinations(v1, v2);

   // Next, the respective elements can be taken via the computed indices.
   auto c1 = Take(v1, idx[0]);
   auto c2 = Take(v2, idx[1]);

   // Finally, you can perform any set of operations conveniently.
   auto v3 = c1 * c2;

   std::cout << "Combinations of " << v1 << " and " << v2 << ":" << std::endl;
   for(size_t i=0; i<v3.size(); i++) {
       std::cout << c1[i] << " * " << c2[i] << " = " << v3[i] << std::endl;
   }
   std::cout << std::endl;

   // However, if you want to compute operations on unique combinations of a
   // single RVec, you can perform this as follows.

   // Get the indices of unique triples for the given vector.
   ROOT::RVecD v4{1., 2., 3., 4.};
   auto idx2 = Combinations(v4, 3);

   // Take the elements and compute any operation on the returned collections.
   auto c3 = Take(v4, idx2[0]);
   auto c4 = Take(v4, idx2[1]);
   auto c5 = Take(v4, idx2[2]);

   auto v5 = c3 * c4 * c5;

   std::cout << "Unique triples of " << v4 << ":" << std::endl;
   for(size_t i=0; i<v4.size(); i++) {
       std::cout << c3[i] << " * " << c4[i] << " * " << c5[i] << " = " << v5[i] << std::endl;
   }
   std::cout << std::endl;
}
