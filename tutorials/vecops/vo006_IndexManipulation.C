/// \file
/// \ingroup tutorial_vecops
/// \notebook -nodraw
/// In this tutorial we demonstrate RVec helpers for index manipulation.
///
/// \macro_code
/// \macro_output
///
/// \date September 2018
/// \author Stefan Wunsch

using namespace ROOT::VecOps;

void vo006_IndexManipulation()
{
   // We assume that we have multiple linked collections, the elements of which
   // represent different objects.
   RVec<float> muon_pt = {20.0, 30.0, 10.0, 25.0};
   RVec<float> muon_eta = {1.0, -2.0, 0.5, 2.5};

   for (size_t i = 0; i < muon_pt.size(); i++) {
      std::cout << "Muon " << i + 1 << " (pt, eta): " << muon_pt[i] << ", "
                << muon_eta[i] << std::endl;
   }

   // First, let's make a selection and write out all indices, which pass.
   auto idx_select = Nonzero(muon_pt > 15 && abs(muon_eta) < 2.5);

   // Second, get indices that sort one of the collections in descending order.
   auto idx_sort = Reverse(Argsort(muon_pt));

   // Finally, we find all indices present in both collections of indices retrieved
   // from sorting and selecting.
   // Note, that the order of the first list passed to the Intersect helper is
   // contained.
   auto idx = Intersect(idx_sort, idx_select);

   // Take from all lists the elements of the passing objects.
   auto good_muon_pt = Take(muon_pt, idx);
   auto good_muon_eta = Take(muon_eta, idx);

   for (size_t i = 0; i < idx.size(); i++) {
      std::cout << "Selected muon " << i + 1 << " (pt, eta): " << good_muon_pt[i]
                << ", " << good_muon_eta[i] << std::endl;
   }
}
