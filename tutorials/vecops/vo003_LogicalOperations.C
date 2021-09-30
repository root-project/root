/// \file
/// \ingroup tutorial_vecops
/// \notebook -nodraw
/// In this tutorial we learn how the RVec class can be used to
/// express logical operations.
///
/// \macro_code
/// \macro_output
///
/// \date May 2018
/// \author Danilo Piparo

void vo003_LogicalOperations()
{

   // Logical operations on RVec instances are made to be very easy to use.
   ROOT::RVecD v1{1., 2., 3.};
   ROOT::RVecD v2{3., 2., 1.};

   // Let's start with operations which act element by element. In this case
   // we expext a RVec which holds {1. > 3., 2. > 2., 3. > 1.}, i.e. {1, 0, 0}:
   auto v1_gr_v2 = v1 > v2;
   std::cout << v1 << " > " << v2 << " = " << v1_gr_v2 << std::endl;

   // Other logical operations are supported, of course:
   auto v1_noteq_v2 = v1 != v2;
   std::cout << v1 << " != " << v2 << " = " << v1_noteq_v2 << std::endl;

   // All returns true if all of the elements equate to true, return false otherwise.
   // Any returns true if any of the elements equates to true, return false otherwise.
   auto all_true = v1 > .5 * v2;
   std::cout << std::boolalpha;
   std::cout << "All( " << v1 << " > .5 * " << v2 << " ) = " << All(all_true) << std::endl;
   std::cout << "Any( " << v1 << " > " << v2 << " ) = " << Any(v1_noteq_v2) << std::endl;

   // Selections on the RVec contents can be applied with the "square brackets" operator,
   // which is not only a way to access the content of the RVec.
   // This operation can change the size of the RVec.
   ROOT::RVecD v{1., 2., 3., 4., 5.};
   auto v_filtered = v[v > 3.];
   std::cout << "v = " << v << ". v[ v > 3. ] = " << v_filtered << std::endl;

   // This filtering operation can be particularely useful when cleaning collections of
   // objects coming from HEP events. For example:
   ROOT::RVecD mu_pt{15., 12., 10.6, 2.3, 4., 3.};
   ROOT::RVecD mu_eta{1.2, -0.2, 4.2, -5.3, 0.4, -2.};

   // Suppose the pts of the muons with a pt greater than 10 and eta smaller than 2.1 are needed:
   auto good_mu_pt = mu_pt[mu_pt > 10 && abs(mu_eta) < 2.1];
   std::cout << "mu_pt = " << mu_pt << "  mu_pt[ mu_pt > 10 && abs(mu_eta) < 2.1] = " << good_mu_pt << std::endl;

   // Advanced logical operations with masking can be performed with the Where helper.
   auto masked_mu_pt = Where(abs(mu_eta) < 2., mu_pt, -999.);
   std::cout << "mu_pt if abs(mu_eta) < 2 else -999 = " << masked_mu_pt << std::endl;
}
