/// \file
/// \ingroup tutorial_vecops
/// \notebook -nodraw
/// In this tutorial we demonstrate RVec helpers for physics computations such
/// as angle differences \f$\Delta \phi\f$.
///
/// \macro_code
/// \macro_output
///
/// \date March 2019
/// \author Stefan Wunsch

using namespace ROOT::VecOps;

void vo007_PhysicsHelpers()
{
   // The DeltaPhi helper computes the closest angle between angles.
   // This means that the resulting value is in the range [-pi, pi].

   // Note that the helper also supports to compute the angle difference of an
   // RVec and a scalar and two scalars. In addition, the computation of the
   // difference and the behaviour at the boundary can be adjusted to radian and
   // degrees.
   RVec<float> phis = {0.0, 1.0, -0.5, M_PI + 1.0};
   auto idx = Combinations(phis, 2);

   auto phi_1 = Take(phis, idx[0]);
   auto phi_2 = Take(phis, idx[1]);
   auto dphi = DeltaPhi(phi_1, phi_2);

   std::cout << "Phi values: " << phis << std::endl;
   for(std::size_t i = 0; i < idx[0].size(); i++) {
      std::cout << "DeltaPhi(" << phis[idx[0][i]] << ", " << phis[idx[1][i]] << ") = " << dphi[i] << std::endl;
   }
}
